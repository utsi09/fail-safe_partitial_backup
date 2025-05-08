#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from carla_msgs.msg import CarlaEgoVehicleStatus
from std_msgs.msg import Bool
from std_msgs.msg import Float64
from warning_mode_interfaces.action import WarningMode
from rclpy.action import ActionServer, CancelResponse, GoalResponse
import carla
import time
import math
import threading
import asyncio
from rclpy.executors import MultiThreadedExecutor

LIDAR_TIMEOUT = 0.5    # 무신호 감지 임계 (초)
CHECK_PERIOD  = 0.1    # 타임아웃 검사 주기 (초)
PUBLISH_RATE  = 10.0   # 제어용 Python API 호출 주기 (Hz)

# 액션 라이브러리 사용해서 behavior Tree로 부터 액션 goal을 받으면 (0, 저속운전 , 1. 갓길 이동 , 2. 차선 평행 회전 , 3. 핸드파킹)

### 위험도 파라미터 #######
K = 3.0 #P에 대한 가중치 ##
lamb = 0.7   # λ      ##
TH = 100              ##
########################

def force_all_traffic_lights_green(client):
    world = client.get_world()
    lights = world.get_actors().filter("traffic.traffic_light")

    for light in lights:
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(9999.0)
        light.freeze(True)
        print(f"신호등 {light.id} → 초록불 고정")


def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

class LidarFailSafe(Node):
    def __init__(self):
        super().__init__('lidar_failsafe')

        # 액션 서버 정의
        self.action_server = ActionServer(
            self,
            WarningMode,
            'warning_mode',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # 액션 실행 상태 플래그
        self.action_running = False
        self.action_future = None
        self.action_goal_handle = None

        # /lidar_alive, /risk_level 퍼블리셔 추가
        self.alive_pub = self.create_publisher(Bool, '/lidar_alive', 10)
        self.risk_pub = self.create_publisher(Float64, '/risk_level', 10)
        
        # ROS: Lidar 구독
        self.create_subscription(
            PointCloud2,
            '/carla/hero/lidar',
            self.lidar_cb,
            10)

        # ROS: 차량 속도(Status) 구독
        self.vehicle_speed = 0.0
        self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/hero/vehicle_status',
            self.status_cb,
            10)

        # CARLA Python API 연결
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        force_all_traffic_lights_green(self.client)  # 강제 초록불

        # HERO 차량 찾기
        self.hero = None
        for v in self.world.get_actors().filter('vehicle.*'):
            if v.attributes.get('role_name') == 'hero':
                self.get_logger().info(f"[DEBUG] 차량 ID={v.id}, role_name={v.attributes.get('role_name')}")
                self.hero = v
                break
        if not self.hero:
            self.get_logger().error('Hero 차량을 찾을 수 없습니다!')

        # 상태 변수 초기화
        self.last_stamp = time.time()
        self.in_fail = False
        self.current_risk = 0.0
        self.waypoint = None
        self.left_lane_marking = None
        self.right_lane_marking = None
        self.left_type = None
        self.right_type = None

        # 타이머 설정
        self.create_timer(CHECK_PERIOD, self.check_timeout)
        self.create_timer(1.0 / PUBLISH_RATE, self.publish_ctrl)  # 안전용 비상 제어 타이머 활성화
        self.create_timer(0.1, self.publish_risk)
        self.create_timer(1.0, self.next_line)

        # 비동기 작업용 루프
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.future = None
        self.loop = None
        self.setup_async_loop()

    def setup_async_loop(self):
        """비동기 루프 설정"""
        self.loop = asyncio.new_event_loop()
        self.executor_thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.executor_thread.start()

    def run_async_loop(self):
        """비동기 루프 실행"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    ## 액션 서버 콜백 함수들 ######################################################################################
    def goal_callback(self, goal_request):
        self.get_logger().info(f'목표 요청 수신: mode={goal_request.mode}')
        
        # 모드 유효성 검사
        valid_modes = [0, 1, 2, 3]
        if goal_request.mode in valid_modes:
            self.action_running = True
            mode_names = ["저속 운전", "갓길 이동", "차선 평행 회전", "핸드파킹"]
            self.get_logger().info(f'목표 승인: {mode_names[goal_request.mode]}')
            return GoalResponse.ACCEPT
        else:
            self.get_logger().warn(f'잘못된 모드: {goal_request.mode}')
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('목표 취소 요청 수신')
        
        # 실행 중인 작업 취소
        if self.action_future and not self.action_future.done():
            self.action_future.cancel()
        
        self.action_running = False
        
        # 차량 안전 정지
        if self.hero:
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, hand_brake=False)
            self.hero.apply_control(ctrl)
            
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        액션 실행 콜백 - 비동기식으로 변경
        """
        self.action_goal_handle = goal_handle
        
        try:
            mode = goal_handle.request.mode
            self.get_logger().info(f'액션 실행: mode={mode}')
            self.action_running = True

            # 오토파일럿 비활성화
            if self.hero:
                self.hero.set_autopilot(False)
            
            # 비동기 작업 시작
            if mode == 0:  # 저속 운전
                result = await self.execute_low_speed_mode(goal_handle)
            elif mode == 1:  # 갓길 이동
                result = await self.execute_shoulder_move_mode(goal_handle)
            elif mode == 2:  # 차선 평행 회전
                result = await self.execute_lane_align_mode(goal_handle)
            elif mode == 3:  # 핸드파킹
                result = await self.execute_hand_parking_mode(goal_handle)
            else:
                self.get_logger().error(f'잘못된 모드: {mode}')
                goal_handle.abort()
                result = WarningMode.Result()
                result.success = False
            
            return result
            
        except asyncio.CancelledError:
            self.get_logger().info('액션이 취소되었습니다')
            goal_handle.canceled()
            result = WarningMode.Result()
            result.success = False
            return result
        except Exception as e:
            self.get_logger().error(f'액션 실행 중 오류 발생: {e}')
            goal_handle.abort()
            result = WarningMode.Result()
            result.success = False
            return result
        finally:
            self.action_running = False
            self.action_goal_handle = None

    # 모드 0: 저속 운전 (비동기식으로 변경)
    async def execute_low_speed_mode(self, goal_handle):
        self.get_logger().info('모드 0: 저속 운전 실행')
        
        # 저속 운전 실행 시간 제한 (20초)
        timeout_time = time.time() + 20.0
        
        # 위험도 감소까지 실행
        while time.time() < timeout_time and not goal_handle.is_cancel_requested:
            if self.hero:
                vel = self.hero.get_velocity()
                # m/s → km/h
                curr_kph = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2)**0.5
                # throttle 입력 범위 
                throttle = max(0.0, min(1.0, 0.6 * (curr_kph / 100.0)))
                
                ctrl = carla.VehicleControl(
                    throttle=float(throttle), # 현재 속도의 60%로 주행
                    steer=0.0,
                    brake=0.0
                )
                self.hero.apply_control(ctrl)
            
            # 피드백 발행
            feedback = WarningMode.Feedback()
            feedback.current_speed = float(self.vehicle_speed)
            goal_handle.publish_feedback(feedback)
            
            # 위험도 로깅
            self.get_logger().info(f"▶ 저속 운전 중")
            
            # 위험도에 따른 종료 조건 체크
            if self.current_risk < 20:
                self.get_logger().info('위험도 감소: 저속 운전 성공적으로 완료')
                goal_handle.succeed()
                break
            elif self.current_risk > 100:
                self.get_logger().warn('위험도 증가: 저속 운전 중단')
                goal_handle.abort()
                break
            
            # 타임아웃 체크
            if time.time() >= timeout_time:
                self.get_logger().warn('저속 운전 시간 초과')
                goal_handle.abort()  # 시간 초과시 실패로 처리
                break
            
            # 0.5초 대기
            await asyncio.sleep(0.5)
        
        # 결과 반환
        result = WarningMode.Result()
        result.success = True
        return result

    # 모드 1: 갓길 이동 (비동기식으로 변경)
    async def execute_shoulder_move_mode(self, goal_handle):
        self.get_logger().info('모드 1: 갓길 이동 실행')
        
        # 갓길 이동 실행 시간 제한 (30초)
        timeout_time = time.time() + 30.0
        reached_shoulder = False
        
        while time.time() < timeout_time and not goal_handle.is_cancel_requested:
            if self.hero:
                # 갓길 도달 여부 확인
                if self.is_on_shoulder():
                    self.get_logger().info('갓길에 도달했습니다')
                    reached_shoulder = True
                    break
                
                # 갓길로 이동하는 적절한 조향 계산
                # 오른쪽 갓길 이동이 기본 동작이므로 오른쪽으로 조향
                ctrl = carla.VehicleControl(throttle=0.3, steer=0.3, brake=0.0)
                self.hero.apply_control(ctrl)
                
                # 피드백
                feedback = WarningMode.Feedback()
                feedback.current_speed = float(self.vehicle_speed)
                goal_handle.publish_feedback(feedback)
        
                self.get_logger().info(f"▶ 갓길로 이동 중")
            
            # 0.5초 대기
            await asyncio.sleep(0.5)
        
        # 결과 처리
        if reached_shoulder or time.time() >= timeout_time:
            goal_handle.succeed()
            result = WarningMode.Result()
            result.success = True
        else:
            goal_handle.abort()
            result = WarningMode.Result()
            result.success = False
        return result

    # 모드 2: 차선 평행 회전 (비동기식으로 변경)
    async def execute_lane_align_mode(self, goal_handle):
        self.get_logger().info('모드 2: 차선 평행 회전 실행')
        
        # 차선 평행 회전 실행 시간 제한 (20초)
        timeout_time = time.time() + 20.0
        aligned = False
        
        while time.time() < timeout_time and not goal_handle.is_cancel_requested:
            if self.hero:
                # 차량과 차선의 yaw 차이 계산
                hero_yaw = self.hero.get_transform().rotation.yaw
                lane_yaw = self.get_lane_rotation()
                angle_diff = abs(normalize_angle(hero_yaw - lane_yaw))
                
                # 피드백
                feedback = WarningMode.Feedback()
                feedback.current_speed = float(self.vehicle_speed)
                goal_handle.publish_feedback(feedback)
                
                # 차선과 평행해지면 성공
                if angle_diff < 5.0:  # 5도 이내면 평행하다고 판단
                    self.get_logger().info('차선과 평행하게 정렬되었습니다')
                    aligned = True
                    # # 평행하게 되면 정차
                    # ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5)
                    # self.hero.apply_control(ctrl)
                    # break
                
                # yaw 차이에 기반한 조향 보정
                steer = max(-1.0, min(1.0, normalize_angle(lane_yaw - hero_yaw) / 45.0))
                # throttle = 0.2 if self.vehicle_speed < 5.0 else 0.0  # 속도가 너무 낮으면 가속
                
                ctrl = carla.VehicleControl(throttle=0.2, steer=steer, brake=0.0)
                self.hero.apply_control(ctrl)
                self.get_logger().info(f"▶ 평행 맞추는 중")
            
            # 0.5초 대기
            await asyncio.sleep(0.5)
        
        # 결과 처리
        if aligned or time.time() >= timeout_time:
            goal_handle.abort()
            result = WarningMode.Result()
            result.success = False
            
        return result

    # 모드 3: 핸드파킹 (비동기식으로 변경)
    async def execute_hand_parking_mode(self, goal_handle):
        self.get_logger().info('모드 3: 핸드파킹 실행')
        
        # 핸드파킹 실행 시간
        parking_time = 9999.0
        start_time = time.time()
        
        while time.time() - start_time < parking_time and not goal_handle.is_cancel_requested:
            if self.hero:
                # 피드백
                feedback = WarningMode.Feedback()
                feedback.current_speed = float(self.vehicle_speed)
                goal_handle.publish_feedback(feedback)
                
                # 핸드브레이크 적용
                ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
                self.hero.apply_control(ctrl)
                self.get_logger().info(f"▶ 핸드브레이크 적용 중 ({time.time() - start_time:.1f}/{parking_time:.1f}초)")
            
            # 차량 정지 확인
            if self.vehicle_speed < 0.1:  # 거의 정지 상태
                self.get_logger().info("차량이 정지했습니다")
                # 핸드브레이크가 적용된 상태로 유지하기 위해 계속 제어 적용
            
            # 0.5초 대기
            await asyncio.sleep(0.5)
        
        # 핸드파킹 완료 후에도 브레이크 상태 유지
        if self.hero:
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
            self.hero.apply_control(ctrl)
            
        # 결과 처리
        goal_handle.succeed()
        result = WarningMode.Result()
        result.success = True
        return result

    # 차량이 갓길에 있는지 확인
    def is_on_shoulder(self):
        if not self.waypoint:
            return False
            
        if (self.left_lane_marking and self.right_lane_marking and 
            self.left_lane_marking.type == carla.LaneMarkingType.Solid and 
            self.right_lane_marking.type == carla.LaneMarkingType.NONE):
            return True
            
        return False

    # 차선의 방향(yaw) 정보를 가져옴
    def get_lane_rotation(self):
        if self.waypoint:
            return self.waypoint.transform.rotation.yaw
        return 0.0

    # 경보 알고리즘 로직 (기존 코드)
    def check_timeout(self):
        t = time.time() - self.last_stamp  # (현재 시간 - 최근 수신 시간)
        alive = (t < LIDAR_TIMEOUT)  # 타임아웃인지 여부 (True/False)
        
        # 위험도 계산 (라이다만 고려)
        self.current_risk = K * math.exp(lamb*t)
        
        # 위험도가 낮으면 간략하게 로그, 위험도가 높으면 좀 더 자세하게 로그
        if self.current_risk < TH:
            self.get_logger().debug(f"● 위험도={self.current_risk:.1f}, LiDAR {'OK' if alive else 'TIMEOUT'}")
        else:
            self.get_logger().warn(f"●●● 위험도={self.current_risk:.1f}, LiDAR {'OK' if alive else 'TIMEOUT'} ●●●")
        
        # 위험도 타임아웃 여부 업데이트
        self.in_fail = not alive
        
        # 위험도 퍼블리시
        msg = Bool()
        msg.data = alive
        self.alive_pub.publish(msg)
    
    # 위험도 발행
    def publish_risk(self):
        risk_msg = Float64()
        risk_msg.data = float(self.current_risk)
        self.risk_pub.publish(risk_msg)
    
    # 차선 정보 업데이트
    def next_line(self):
        if not self.hero:
            return
        
        try:
            self.waypoint = self.world.get_map().get_waypoint(
                self.hero.get_location(), 
                project_to_road=True, 
                lane_type=carla.LaneType.Any
            )
            
            self.left_lane_marking = self.waypoint.left_lane_marking
            self.right_lane_marking = self.waypoint.right_lane_marking
            
            # 차선 타입 업데이트
            self.left_type = self.left_lane_marking.type if self.left_lane_marking else None
            self.right_type = self.right_lane_marking.type if self.right_lane_marking else None
            
        except Exception as e:
            self.get_logger().error(f"차선 정보 업데이트 오류: {e}")
    
    # 비상 상황 제어 (액션이 실행중이 아닐 때만)
    def publish_ctrl(self):
        # 액션이 실행 중이거나 비상 상황이 아니면 무시
        if self.action_running or not self.in_fail or not self.hero:
            return
            
        # 위험도가 TH를 넘으면 자동 대응
        if self.current_risk >= TH:
            self.get_logger().warn("▶ 위험 상황: 자동 감속 적용")
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5)
            self.hero.apply_control(ctrl)
    
    # 차량 상태 업데이트
    def status_cb(self, msg):
        # m/s 단위로 저장
        self.vehicle_speed = msg.velocity
    
    # 라이다 수신 콜백
    def lidar_cb(self, msg):
        # 라이다 메시지 수신 시점 갱신
        self.last_stamp = time.time()
        
        # alive 토픽에 True 발행
        alive_msg = Bool()
        alive_msg.data = True
        self.alive_pub.publish(alive_msg)
        
    def __del__(self):
        """소멸자: 리소스 정리"""
        if self.loop and self.loop.is_running():
            self.loop.stop()
        if hasattr(self, 'executor_thread') and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = LidarFailSafe()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()