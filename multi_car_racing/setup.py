from setuptools import setup

setup(
    name='gym_multi_car_racing',
    version='1.0.1',
    url='https://github.com/igilitschenski/multi_car_racing',
    description='Gym Multi Car Racing Environment',
    packages=['gym_multi_car_racing'],
    install_requires=[
    "pyglet",
    "shapely",
    "gymnasium[Box2d]"

    ]
)
