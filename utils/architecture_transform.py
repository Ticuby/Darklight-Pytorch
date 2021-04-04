

from .video_transforms import Normalize, ToTensor2, ToTensor, Scale, Compose
def determine_architecture_transform(architecture_name_list, num_seg, length):
    transform_list = []
    for architecture_name in architecture_name_list:
        clip_mean = [0.43216, 0.394666, 0.37645] * num_seg * length
        clip_std = [0.22803, 0.22145, 0.216989] * num_seg * length

        scale = 0.5
        size = 112
            
        normalize = Normalize(mean=clip_mean, std=clip_std)  
        scale_transform = Scale(size)

        tensor_transform = ToTensor()
            
        transform = Compose([
                scale_transform,
                tensor_transform,
                normalize,
            ])
        transform_list.append(transform)
        print(architecture_name)
        print('size: %d' %(size))
    return transform_list


def determine_architecture_transform2(architecture_name_list, num_seg, length):
    transform_list = []
    for architecture_name in architecture_name_list:
        clip_mean = [0.43216, 0.394666, 0.37645, 0.5, 0.5] * num_seg * length
        clip_std = [0.22803, 0.22145, 0.216989, 0.225, 0.225] * num_seg * length

        scale = 0.5
        size = 112
            
        normalize = Normalize(mean=clip_mean, std=clip_std)  
        scale_transform = Scale(size)
        tensor_transform = ToTensor()
            
        transform = Compose([
                scale_transform,
                tensor_transform,
                normalize,
            ])
        transform_list.append(transform)
        print(architecture_name)
        print('size: %d' %(size))
    return transform_list
        
