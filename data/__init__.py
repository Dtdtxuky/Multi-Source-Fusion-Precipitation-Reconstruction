'''create dataset and dataloader'''
import torch.utils.data

def create_dataloader(dataset, dataset_opt, phase, sampler=None):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=False,
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            sampler=sampler)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    from data.config_v2 import cfg

    # 拼图测试集-输入的是大图的地址，数据集的创建首先将其分成patch，将小图的结果保存
    if dataset_opt == 'patch2all':
        print('Using patch2all')
        from data.InfData_patch2all import CMA_Dataset
        dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/Inf_allpath.txt', phase='train')
        
    # patch测试数据集-输入的是patch的地址以及对应的x、y起始坐标
    if dataset_opt == 'Dif_patch':
        print('Using Dif_patch')
        from data.Dif_data import CMA_Dataset 
        if phase == 'train':
            dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/Train_path_3years_pre_th1_5_th2_500_radar_th1_3_th2_400.txt', phase=phase)
            print(len(dataset)) 
        else:
            dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/TestSetting3.txt', phase='train')
            
    # 训练Dif模型数据集      
    if dataset_opt == 'Dif':
        print('Using Dif')
        from data.Dif_data import CMA_Dataset 
        if phase == 'train':
            dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/Train_path_3years_pre_th1_5_th2_500_radar_th1_3_th2_400.txt', phase=phase)
            print(len(dataset)) 
        else:
            dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/TestSetting3.txt', phase='train')

    # 训练AE的数据集
    if dataset_opt == 'AE':
        print('Using AE Data')
        from data.AE_data import CMA_Dataset 
        if phase == 'train':
            dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/AE_path.txt', phase=phase)
        else:
            dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/AE_path.txt', phase='train')
    
    return dataset
