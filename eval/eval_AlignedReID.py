from eval.metric_AlignedReID import cmc, mean_ap
import scipy.io as sio

import pdb
import numpy as np

dataset = 'cuhk03'
ckp_epoch = 200
mat_path = 'save_mat_reranking_7_3_0p85/'

if dataset == 'cuhk03':
    separate_camera_set=True
    single_gallery_shot=True
    first_match_break=False


def eval_map_cmc(
      q_g_dist,
      q_ids=None, g_ids=None,
      q_cams=None, g_cams=None,
      separate_camera_set=None,
      single_gallery_shot=None,
      first_match_break=None,
      topk=None):
    """Compute CMC and mAP.
    Args:
      q_g_dist: numpy array with shape [num_query, num_gallery], the
        pairwise distance between query and gallery samples
    Returns:
      mAP: numpy array with shape [num_query], the AP averaged across query
        samples
      cmc_scores: numpy array with shape [topk], the cmc curve
        averaged across query samples
    """
    # Compute mean AP
    mAP = mean_ap(
      distmat=q_g_dist,
      query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams)
    # Compute CMC scores
    cmc_scores = cmc(
      distmat=q_g_dist,
      query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams,
      separate_camera_set=separate_camera_set,
      single_gallery_shot=single_gallery_shot,
      first_match_break=first_match_break,
      topk=topk)
    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}], [cmc20: {:5.2%}]'
          .format(mAP, *cmc_scores[[0, 4, 9, 19]]))
    return mAP, cmc_scores


def main():

    dismat = sio.loadmat(mat_path + 'dismat{}.mat'.format(ckp_epoch))
    dismat = dismat['dismat']
    g_pids = sio.loadmat(mat_path + 'g_pids{}.mat'.format(ckp_epoch))
    g_pids = np.squeeze(np.transpose(g_pids['g_pids']))
    q_pids = sio.loadmat(mat_path + 'q_pids{}.mat'.format(ckp_epoch))
    q_pids = np.squeeze(np.transpose(q_pids['q_pids']))
    g_camids = sio.loadmat(mat_path + 'g_camids{}.mat'.format(ckp_epoch))
    g_camids = np.squeeze(np.transpose(g_camids['g_camids']))
    q_camids = sio.loadmat(mat_path + 'q_camids{}.mat'.format(ckp_epoch))
    q_camids = np.squeeze(np.transpose(q_camids['q_camids']))


    eval_map_cmc(
        dismat,
        q_ids=q_pids, g_ids=g_pids,
        q_cams=q_camids, g_cams=g_camids,
        separate_camera_set=separate_camera_set,
        single_gallery_shot=single_gallery_shot,
        first_match_break=first_match_break,
        topk=20)


if __name__ == '__main__':
    main()