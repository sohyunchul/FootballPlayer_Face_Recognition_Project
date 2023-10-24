from roboflow import Roboflow
rf = Roboflow(api_key="XhEv2rhQgUdDCURAOLnb")
project = rf.workspace("sohyunchul").project("opencvfootball")
dataset = project.version(2).download("yolov8")
