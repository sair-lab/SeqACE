import os


if __name__ == '__main__':
    dataroot = "/home/xukuan/project/seq_ace/seq_scr/datasets"
    model_dir = "/home/xukuan/project/seq_ace/seq_scr/output"

    # sequences = ['pgt_7scenes_chess', 'pgt_7scenes_heads', 'pgt_7scenes_pumpkin', 'pgt_7scenes_fire', 'pgt_7scenes_office', 'pgt_7scenes_redkitchen', 'pgt_7scenes_stairs']
    sequences = ['pgt_7scenes_office']
    # sequences = ['Cambridge_GreatCourt', 'Cambridge_KingsCollege', 'Cambridge_OldHospital', 'Cambridge_ShopFacade', 'Cambridge_StMarysChurch']
    for seq in sequences:
        seq_data_root = os.path.join(dataroot, seq)
        model_path = os.path.join(model_dir, seq + ".pt")
        os.system("python test_ace_feature.py {} {}".format(seq_data_root, model_path))