from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait

ev3 = EV3Brick()

motor_links = Motor(Port.B)
motor_rechts = Motor(Port.C)

while True:
    motor_links.run(300)
    motor_rechts.run(300)
    wait(100)
