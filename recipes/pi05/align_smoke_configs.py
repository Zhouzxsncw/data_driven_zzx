from pathlib import Path
import shutil,time,yaml
root=Path('/home/tk/data_Driven')
collect=yaml.safe_load(Path('/home/tk/infer/config/dual_arm_gripper_2cam.yaml').read_text())
stamp=time.strftime('%Y%m%d_%H%M%S')
for step in (1302,1674):
 p=root/f'infer/infer/config/lerobot/pi05_00117_compare_step{step:06d}_tk_ws.yaml'
 shutil.copy2(p,str(p)+'.before_smoke_'+stamp)
 cfg=yaml.safe_load(p.read_text())
 current={d['serial']:d for d in cfg['camera_sender']['devices']}
 ordered=[]
 for d in collect['cameras']['devices']:
  item=current[d['serial']]
  for key in ('width','height','fps'):item[key]=d[key]
  ordered.append(item)
 cfg['camera_sender']['devices']=ordered
 cfg['exo_safety']['collect_config']=str(root/'infer/data_collect/config/dual_arm_gripper_2cam_tk.yaml')
 p.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
 print('Aligned',p)
