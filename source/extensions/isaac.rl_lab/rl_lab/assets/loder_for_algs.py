from .base_motionloader import MotionData_Base

class AmpMotion(MotionData_Base):
    def __init__(self, 
                 data_dir,
                 datatype="isaaclab",
                 file_type="csv",
                 data_spaces = None,
                 env_step_duration = 0.005,**kwargs):
        super().__init__(data_dir,datatype,file_type,data_spaces,env_step_duration,**kwargs)
        self.prepare_amp_state_trans()
    def prepare_amp_state_trans(self):
        self.amp_state_trans = []
        amp_data_spaces  ={
            "joint_pos",
            "foot_pos",
            "base_lin_vel",
            "base_ang_vel",
            "joint_vel",
            "z_pos"}
        for tragedy in self.data_tensors:
            
            
            
            s = tragedy[:-1]    
            s_next = tragedy[1:]
    
        
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        
        pass
    #TODO