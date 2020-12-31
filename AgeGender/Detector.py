from AgeGender.models.model import ShuffleneTiny, ShuffleneFull
import torch


class AgeGender:

    def __init__(self, name, weight_path, device):
        """
        Age and gender Detector
        :param name: name of backbone (full or tiny)
        :param device: model run in cpu or gpu (cuda, cpu)
        :param weight_path: path of network weight

        Notice: image size must be 112x112
        but cun run with 224x224

        Method detect:
                :param faces: 4d tensor of face for example size(1, 3, 112, 112)
                :returns genders list and ages list
        """
        if name == 'tiny':
            model = ShuffleneTiny()
        elif name == 'full':
            model = ShuffleneFull()
        else:
            exit('from AgeGender Detector: model dose not support just(tiny, full)')

        model.load_state_dict(torch.load(weight_path))
        model.to(device).eval()
        self.model = model
        self.device = device

    def detect(self, faces):
        faces = faces.permute(0, 3, 1, 2)
        faces = faces.float().div(255).to(self.device)

        mu = torch.as_tensor([0.485, 0.456, 0.406], dtype=faces.dtype, device=faces.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=faces.dtype, device=faces.device)
        faces[:].sub_(mu[:, None, None]).div_(std[:, None, None])

        outputs = self.model(faces)
        genders = []
        ages = []
        for out in outputs:
            gender = torch.argmax(out[:2])
            gender = 'Male' if gender == 0 else 'Female'
            genders.append(gender)
            ages.append(int(out[-1]))

        return genders, ages
