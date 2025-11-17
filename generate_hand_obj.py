from manopth.manolayer import ManoLayer
import torch
import numpy as np
import trimesh
import json
import os
from tqdm import tqdm

# reference hand obj to see if obj is generated correctly
# boardgame_v_W_qdSiPKSdQ_frame000019_hand.obj

# define paths
output_obj_dir = "/restricted/projectnb/cs599dg/mwakeham/mcc-ho/mow_data/hand_models"
mano_model_dir = "/restricted/projectnb/cs599dg/mwakeham/mcc-ho/_DATA/data/mano"
mano_poses_json_path = "/restricted/projectnb/cs599dg/mwakeham/mcc-ho/mow_data/poses.json"

# intiialize mano model
mano_layer = ManoLayer(
    mano_root=mano_model_dir,
    use_pca=False,
    ncomps=45,
    flat_hand_mean=True
)

# load mano pose data from mow
with open(mano_poses_json_path, 'r') as f:
    mano_poses = json.load(f)

# save mano .obj files using mano pose data
for entry in tqdm(mano_poses):
    pose = torch.tensor(entry["hand_pose"], dtype=torch.float32).unsqueeze(0)
    betas = torch.zeros(1, 10)

    hand_t = torch.tensor(entry["hand_t"], dtype=torch.float32).unsqueeze(0)
    hand_R = torch.tensor(entry["hand_R"], dtype=torch.float32).reshape(1, 3, 3)
    hand_s = entry["hand_s"]

    verts, _ = mano_layer(pose, betas)
    verts_scaled = hand_s * verts
    verts_rotated = torch.matmul(verts_scaled, hand_R.transpose(1, 2))
    verts_world = verts_rotated + hand_t

    faces = mano_layer.th_faces.numpy()
    verts_np = verts_world.squeeze().numpy()

    filename = os.path.basename(entry["obj_url"])
    output_path = os.path.join(output_obj_dir, filename)
    trimesh.Trimesh(vertices=verts_np, faces=faces).export(output_path)

# compare with gt .obj file to see if same
gt_obj_path = "/restricted/projectnb/cs599dg/mwakeham/mcc-ho/demo/boardgame_v_W_qdSiPKSdQ_frame000019_hand.obj"
generated_obj_path = os.path.join(output_obj_dir, "boardgame_v_W_qdSiPKSdQ_frame000019.obj")

def load_obj_vertices_faces(path):
    verts, faces = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                faces.append([
                    int(i.split('/')[0]) for i in line.strip().split()[1:]
                ])
    return np.array(verts), np.array(faces)

v1, f1 = load_obj_vertices_faces(gt_obj_path)
v2, f2 = load_obj_vertices_faces(generated_obj_path)

same_verts = np.allclose(v1, v2, atol=1e-6)
same_faces = np.array_equal(f1, f2)

print("identical vertices:", same_verts)
print("identical faces:", same_faces)