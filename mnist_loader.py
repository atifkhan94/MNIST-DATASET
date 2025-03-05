import numpy as np
import struct

def read_idx(filename):
    """Read IDX file format used by MNIST."""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist():
    """Load MNIST training and testing data."""
    train_images = read_idx('train-images.idx3-ubyte')
    train_labels = read_idx('train-labels.idx1-ubyte')
    test_images = read_idx('t10k-images.idx3-ubyte')
    test_labels = read_idx('t10k-labels.idx1-ubyte')
    
    # Normalize the images to [0, 1] range
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

if __name__ == '__main__':
    # Example usage
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    print(f'Training images shape: {train_images.shape}')
    print(f'Training labels shape: {train_labels.shape}')
    print(f'Test images shape: {test_images.shape}')
    print(f'Test labels shape: {test_labels.shape}')