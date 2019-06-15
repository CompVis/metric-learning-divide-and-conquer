from .base import *

class SOProducts(BaseDataset):
    nb_train_all = 59551
    nb_test_all = 60502
    def __init__(self, root, classes, transform=None):
        BaseDataset.__init__(self, root, classes, transform)

        classes_train = range(0, 11318)
        classes_test = range(11318, 22634)

        if classes.start in classes_train:
            if classes.stop - 1 in classes_train:
                train = True

        if classes.start in classes_test:
            if classes.stop - 1 in classes_test:
                train = False

        with open(
            os.path.join(
            root,
            'Ebay_{}.txt'.format('train' if train else 'test')
            )
        ) as f:

            f.readline()
            index = 0
            nb_images = 0

            for (image_id, class_id, _, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1

            if train:
                assert nb_images == type(self).nb_train_all
            else:
                assert nb_images == type(self).nb_test_all
