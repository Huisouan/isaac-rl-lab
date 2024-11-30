from .base_motionloader import MotionData_Base

class AmpMotion(MotionData_Base):
    def __init__(self, 
                 data_dir,
                 datatype="isaaclab",
                 file_type="csv",
                 data_spaces = None,
                 env_step_duration = 0.005,**kwargs):
        super().__init__(data_dir,datatype,file_type,data_spaces,env_step_duration,**kwargs)
        
    def 