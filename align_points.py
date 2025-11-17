import open3d as o3d
import numpy as np

# source_path = "/Users/michaelwakeham/project/panorama/supplement+bottle+3d+model_source_red.ply"
source_path = "/Users/michaelwakeham/project/panorama/bleach+bottle+3d+model_source_blue.ply"

target_path = "/Users/michaelwakeham/project/panorama/021_bleach_cleanser_textured_simple_target_green.ply"

pcd_source = o3d.io.read_point_cloud(source_path)
pcd_target = o3d.io.read_point_cloud(target_path)

src_pts = np.asarray(pcd_source.points)
tgt_pts = np.asarray(pcd_target.points)

src_centered = src_pts - src_pts.mean(axis=0)
tgt_centered = tgt_pts - tgt_pts.mean(axis=0)

scale_factor = np.mean(np.linalg.norm(tgt_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))
pcd_source.scale(scale_factor, center=pcd_source.get_center())
print("Applied scale factor:", scale_factor)

voxel_size = 0.005
pcd_source_down = pcd_source.voxel_down_sample(voxel_size)
pcd_target_down = pcd_target.voxel_down_sample(voxel_size)

pcd_source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
pcd_target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

threshold = 0.02  # max correspondence distance
trans_init = np.eye(4)  # initial guess

reg_p2l = o3d.pipelines.registration.registration_icp(
    pcd_source_down, pcd_target_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("ICP matrix")
print(reg_p2l.transformation)

pcd_source.transform(reg_p2l.transformation)

o3d.visualization.draw_geometries([pcd_source, pcd_target])

aligned_path = source_path.replace(".ply", "_aligned.ply")
o3d.io.write_point_cloud(aligned_path, pcd_source)


def compute_fscore(pcd_pred, pcd_gt, threshold):
    pts_pred = np.asarray(pcd_pred.points)
    pts_gt = np.asarray(pcd_gt.points)

    tree_pred = o3d.geometry.KDTreeFlann(pcd_pred)
    tree_gt = o3d.geometry.KDTreeFlann(pcd_gt)

    d_pred = []
    for p in pts_pred:
        [_, idx, _] = tree_gt.search_knn_vector_3d(p, 1)
        q = pts_gt[idx[0]]
        d_pred.append(np.linalg.norm(p - q))
    precision = np.mean(np.array(d_pred) < threshold)

    d_gt = []
    for q in pts_gt:
        [_, idx, _] = tree_pred.search_knn_vector_3d(q, 1)
        p = pts_pred[idx[0]]
        d_gt.append(np.linalg.norm(q - p))
    recall = np.mean(np.array(d_gt) < threshold)

    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall

for tau in [0.005, 0.010, 0.015]:
    f1, precision, recall = compute_fscore(pcd_source, pcd_target, threshold=tau)
    print(f"\nF-score @ {tau*1000:.0f} mm:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {f1:.4f}")
