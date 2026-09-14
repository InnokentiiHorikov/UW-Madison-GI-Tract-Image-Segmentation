import torchio as tio 
from torch import Generator

class FixingAxes(tio.IntensityTransform):
    """
    The purpose of this class is fixing axes of torchio subject,
    because somehow torchio subject mistakes depth and channel axes 
    """
    def __init__(self):
        super().__init__()

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        #D, C, H, W -> C, H, W, D
        for image in subject.get_images(intensity_only=False):
            image.data = image.data.permute(1,2,3,0)
        return subject


def creating_patch_loader(object_path: str,
                          label_path: str,
                          patch_size: (int, int, int),
                          seed: int = 42) -> torchio.SubjectLoader():
    
    tio_subjects = [tio.Subject(image=tio.ScalarImage(object_path),
                segmentation=tio.LabelMap(label_path)) 
                for object_path, label_path in zip(objects_train, masks_train)]

    transform = tio.Compose([FixingAxes()])

    subjects_dataset = tio.SubjectsDataset(tio_subjects, 
                                             transform = transform)

    queue = tio.Queue(
                    subjects_dataset=subjects_dataset,
                    max_length=200,           # Maximum number of patches in the queue
                    samples_per_volume=10,    # Number of patches extracted per volume per loading
                    sampler=sampler,
                    num_workers=2,            # Number of worker threads for loading
            )    

    g = Generator()
    g.manual_seed(42)        
                           
    patches_loader = tio.SubjectsLoader(
                        queue,
                        batch_size=8,
                        shuffle=True,          # Enables shuffling
                        generator=g
                    )
    return patches_loader
                        
