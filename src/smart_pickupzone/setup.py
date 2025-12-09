import os
from pathlib import Path
from setuptools import find_packages, setup

package_name = 'smart_pickupzone'
here = Path(__file__).resolve().parent
package_dir = here / package_name
model_dir = package_dir / 'model'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# 모델 파일을 설치 경로에 포함시켜 런타임에 접근 가능하도록 처리
for root, _, files in os.walk(model_dir):
    if not files:
        continue
    rel_dir = Path(root).relative_to(package_dir)
    install_dir = Path('share') / package_name / rel_dir
    # setup.py가 있는 위치를 기준으로 한 상대경로를 지정해야 colcon이 허용함
    file_paths = [str(Path(package_name) / rel_dir / f) for f in files]
    data_files.append((str(install_dir), file_paths))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'cv_bridge',
        'onnxruntime',
        'numpy',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='jun',
    maintainer_email='snowpoet@naver.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rtdetr_v2_test_node = smart_pickupzone.rtdetr_v2_test_node:main',
            'rtdetr_v4_test_node = smart_pickupzone.rtdetr_v4_test_node:main',
        ],
    },
)
