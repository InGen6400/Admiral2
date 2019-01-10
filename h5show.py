import h5py

H5PATH = 'model.h5'

def traverse(node, depth):
    def indent(depth):
        # 階層に応じて、インデントする。
        for i in range(depth):
            print('    ', end='')

    indent(depth)
    if isinstance(node, h5py.File):
        print('File key:{}'.format(node.name))
    elif isinstance(node, h5py.Group):
        print('Group key:{}'.format(node.name))
    else:
        print('Dataset key:{}, type: {}, shape: {}'.format(
            node.name, node.dtype, node.shape))

    # attributes
    for key, value in node.attrs.items():
        indent(depth)
        print('attribute key: {}, value: {}'.format(key, value))

    # Traverse children
    if not isinstance(node, h5py.Dataset):
        for key, value in node.items():
            traverse(value, depth + 1)


h5file = h5py.File(H5PATH, 'r')
traverse(h5file, 0)