def get_sr_and_score(imset, model, aposterior_gt, num_frames, min_L=16):
    '''
    Super resolves an imset with a given model.
    Args:
        imset: imageset
        model: HRNet, pytorch model
        min_L: int, pad length
    Returns:
        sr: tensor (1, C_out, W, H), super resolved image
        scPSNR: float, shift cPSNR score
    '''

    if imset.__class__ is ImageSet:
        collator = collateFunction(num_frames, min_L)
        lrs, alphas, hrs, hr_maps, names = collator([imset])
    elif isinstance(imset, tuple):  # imset is a tuple of batches
        lrs, alphas, hrs, hr_maps, names = imset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # alphas = torch.from_numpy(np.zeros((1, min_L)))  # torch.tensor
    #    sr = np.zeros((imset[0].shape[0] * 3, imset[0].shape[1] * 3, 3))

    #    for i in range(3):
    #        cur_lrs = np.zeros((1, min_L, imset[0].shape[0], imset[0].shape[1]))

    #        for j in range(min_L):
    #            cur_lrs[0][j] = imset[j][:, :, i]

    #        cur_lrs = torch.from_numpy(cur_lrs)
    #        cur_lrs = cur_lrs.float().to(device)

    #        cur_sr = model(cur_lrs, alphas)[:, 0]
    #        cur_sr = cur_sr.detach().cpu().numpy()[0]

    #        sr[:, :, i] = cur_sr[:, :]

    lrs = lrs[:, :min_L, :, :].float().to(device)
    alphas = alphas[:, :min_L].float().to(device)

    print("LRS SHAPEE: ", lrs.shape)
    print("ALPHAS SHAPEE: ", alphas.shape)

    sr = model(lrs, alphas)[:, 0]
    sr = sr.detach().cpu().numpy()[0]

    if len(hrs) > 0:
        scPSNR = shift_cPSNR(sr=np.clip(sr, 0, 1),
                             hr=hrs.numpy()[0],
                             hr_map=hr_maps.numpy()[0])
    else:
        scPSNR = None

    ssim = cSSIM(sr=np.clip(sr, 0, 1), hr=hrs.numpy()[0])

    #   print("APGT SHAPE: ", aposterior_gt.shape)
    #   print("APGT: ", aposterior_gt)

    if (str(type(aposterior_gt)) == "<class 'NoneType'>"):
        aposterior_ssim = 1.0
    else:
        aposterior_ssim = cSSIM(sr=np.clip(sr, 0, 1), hr=np.clip(aposterior_gt, 0, 1))

    return sr, scPSNR, ssim, aposterior_ssim