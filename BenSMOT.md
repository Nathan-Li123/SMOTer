# BenSMOT
#### Organization
The proposed BenSMOT is organized as follows:
```text
BenSMOT
├── train
|   └──dancing_with_child
|      └──01Ss_Kfo6ag_0
|         └──gt
|            └──gt.txt
|         └──imgs
|            └──001.jpg
|            └──002.jpg
|            └──003.jpg
|            ...
|         └──video_caption.txt
|         └──instance_captions.txt
|         └──interactions.graphml
|      └──0B1uJGWKxCg_0
|      ...
|   └──playing_basketball
|   ...
├── test
|   ...
```
#### Format of Each Video Sequence
For each video sequence, we provide the ground truth files in the `gt` folder (using the standard MOTChallenge annotation format) and the sequence images in the `imgs` folder. In the `video_caption.txt`, `instance_captions.txt`, and `interactions.graphml` files, we provide the overall caption of the video sequence, the caption for each trajectory, and the relationships between the trajectories, respectively. In these files, terms like man0, woman0, man1, etc., are used to represent the trajectories. The order of their appearance corresponds to the trajectory numbers. 

For example, in the following `instance_captions.txt` file, the three trajectories represented by woman0, woman1, and man0 are numbered 1, 2, and 3, respectively, in the corresponding `gt.txt` file.
```text
woman0: A woman in the red blouse covers her face with her hands to...
woman1: A woman in pink short-sleeved shirt and white cap throws water...
man0: A man in yellow sleeves walks behind the woman in pink sleeves ...
```