from .base import *

class VehicleID(BaseDataset):
    def __init__(self, root, classes, transform = None):
        BaseDataset.__init__(self, root, classes, transform)
        # amount of images deviates slightly from what's reported online

        if classes == range(0, 13164):
            fname = 'train_list.txt'
        elif classes == range(13164, 13164 + 800):
            fname = 'test_list_800.txt'
        elif classes == range(13164, 13164 + 1600):
            fname = 'test_list_1600.txt'
        elif classes == range(13164, 13164 + 2400):
            fname = 'test_list_2400.txt'
        else:
            print('Unknown range for classes selected')
            input()

        with open(
            os.path.join(root, 'train_test_split', fname), 'r'
        ) as f:
            lines = [l.strip('\n').split(' ') for l in f.readlines()]

        i = 0
        for l in lines:
            self.im_paths += [os.path.join(root, 'image', l[0] + '.jpg')]
            self.ys += [int(l[1])]
            self.I += [i]
            i += 1

        idx_to_class = {idx: i for i, idx in enumerate(
            sorted(set(self.ys))
        )}
        self.ys = list(
            map(lambda x: idx_to_class[x], self.ys))

    def nb_classes(self):
        assert len(set(self.ys)) == len(set(self.classes))
        return len(self.classes)
