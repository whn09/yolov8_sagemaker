import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/DEMO-pytorch-yolov8-classifier'

role = sagemaker.get_execution_role()

s3_dir = 's3://datalab/data/minc-2500-tiny/'
inputs = {'train': s3_dir+'train', 'test': s3_dir+'val'}
print(inputs)

from sagemaker.pytorch import PyTorch

hyperparameters = {'data': '/opt/ml/input/data',  # cifar10, cifar100, mnist, mnist-fashion, or a directory /opt/ml/input/data
                   'weights': 'yolov8x-cls.pt',  # 'yolov8s-cls.pt'
                   'project': '/opt/ml/checkpoints/',  # '/opt/ml/model/'
                   'name': 'tutorial', 'imgsz': 224, 'batch': 128, 'epochs': 50, 'save-period': 1}  # Single CPU or GPU
#                    'name': 'tutorial', 'imgsz': 224, 'batch': 16*8, 'epochs': 5, 'save-period': 1, 'device': '0,1,2,3,4,5,6,7'}  # Multi-GPU: DP Mode

instance_type = 'ml.g5.2xlarge'  # 'ml.p3.2xlarge' or 'ml.p3.8xlarge' or ...

checkpoint_in_bucket="checkpoints"
# The S3 URI to store the checkpoints
checkpoint_s3_bucket="s3://{}/{}/{}".format(bucket, prefix, checkpoint_in_bucket)
# The local path where the model will save its checkpoints in the training container
checkpoint_local_path="/opt/ml/checkpoints"

estimator = PyTorch(entry_point='train.py',
                            source_dir='./code/',
                            role=role,
                            hyperparameters=hyperparameters,
                            framework_version='2.6.0',  # '1.13.1', '2.5.1'
                            py_version='py312',  # 'py39', 'py311'
                            script_mode=True,
                            instance_count=1,  # 1 or 2 or ...
                            instance_type=instance_type,
                            checkpoint_s3_uri=checkpoint_s3_bucket,
                            checkpoint_local_path=checkpoint_local_path)

estimator.fit(inputs)