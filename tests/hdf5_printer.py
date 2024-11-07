#%%
import h5py

def print_hdf5_contents(file_path, max_elements=10):
    """
    Print the contents of an HDF5 file, including data inside datasets.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    max_elements : int, optional
        Maximum number of elements to print for each dataset (default is 10).
    """
    def print_attrs(name, obj):
        print(f"Name: {name}")
        for key, val in obj.attrs.items():
            print(f"    Attribute: {key} => {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"    Dataset shape: {obj.shape}")
            print(f"    Dataset dtype: {obj.dtype}")
            # Print dataset data, limited to max_elements
            data = obj[()]
            if data.size > max_elements:
                print(f"    Data (first {max_elements} elements): {data.flat[:max_elements]}")
            else:
                print(f"    Data: {data}")
        elif isinstance(obj, h5py.Group):
            print(f"    Group with {len(obj)} members")
        
        # Check if the name contains "results"
        if "results" in name.lower():
            print(f"    *** This name contains 'results' ***")
            if isinstance(obj, h5py.Dataset):
                print(f"    Full data: {data}")
            elif isinstance(obj, h5py.Group):
                print(f"    Group members: {list(obj.keys())}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)

# Example usage
file_path = r"C:\Users\exp4-ArVe-220\Desktop\Evaluix2\tests\TestSaving2.h5"
print_hdf5_contents(file_path)