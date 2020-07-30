


def collateFunction(self, batch):
    """
    Custom collate function to adjust a variable number of low-res images.
    Args:
        batch: list of imageset
    Returns:
        padded_lr_batch: tensor (B, min_L, W, H), low resolution images
        alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
        hr_batch: tensor (B, W, H), high resolution images
        hm_batch: tensor (B, W, H), high resolution status maps
        isn_batch: list of imageset names
    """

    lr_batch = []  # batch of low-resolution views
    alpha_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
    hr_batch = []  # batch of high-resolution views
    hm_batch = []  # batch of high-resolution status maps
    isn_batch = []  # batch of site names

    train_batch = True

    for imageset in batch:
        lrs = imageset['lr']
        L, H, W = lrs.shape
        lr_batch.append(lrs)
        alpha_batch.append(torch.ones(L))
        hr = imageset['hr']
        hr_batch.append(hr)
        hm_batch.append(imageset['hr_map'])
        isn_batch.append(imageset['name'])

    padded_lr_batch = lr_batch
    padded_lr_batch = torch.stack(padded_lr_batch, dim=0)
    alpha_batch = torch.stack(alpha_batch, dim=0)
    hr_batch = torch.stack(hr_batch, dim=0)
    hm_batch = torch.stack(hm_batch, dim=0)
    #        isn_batch = torch.stack(isn_batch, dim=0)

    #       for imageset in batch:#

    #           lrs = imageset['lr']
    #           L, H, W = lrs.shape

    #           if L >= self.min_L:  # pad input to top_k
    #               lr_batch.append(lrs[:self.min_L])
    #               alpha_batch.append(torch.ones(self.min_L))
    #           else:
    #               pad = torch.zeros(self.min_L - L, H, W)
    #               lr_batch.append(torch.cat([lrs, pad], dim=0))
    #               alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

    #           hr = imageset['hr']
    #           if train_batch and hr is not None:
    #               hr_batch.append(hr)
    #           else:
    #               train_batch = False

    #           hm_batch.append(imageset['hr_map'])
    #           isn_batch.append(imageset['name'])

    #       padded_lr_batch = torch.stack(lr_batch, dim=0)
    #       alpha_batch = torch.stack(alpha_batch, dim=0)

    #      if train_batch:
    #          hr_batch = torch.stack(hr_batch, dim=0)
    #          hm_batch = torch.stack(hm_batch, dim=0)

    return padded_lr_batch, alpha_batch, hr_batch, hm_batch, isn_batch