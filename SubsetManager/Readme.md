# SubsetManager

## Usage

Below are examples demonstrating how to use the `SubsetManager` class in a Python script.

### Importing SubsetManager

First, import the `SubsetManager` class:

```python
from SubsetManager.SubsetManager import SubsetManager
```

### Creating a Subset

You can create a subset for a specified supercategory. If no name is provided, the subset will be automatically named based on the supercategory and sampling method. Here's how you can create a subset:

```python
s = SubsetManager()
subset_name = s.create_subset("Mammals", size=10)
print(f"Created subset: {subset_name}")
```

**Output:**

```
Created subset: Mammals_Random_1
```

#### Specifying a Custom Subset Name

You can also specify a custom name for your subset:

```python
custom_subset_name = s.create_subset("Mammals", name="MyMammalsSubset", size=5)
print(f"Created subset: {custom_subset_name}")
```

**Output:**

```
Created subset: Mammals_MyMammalsSubset
```

**Directory Naming Convention:**

- Default: `<supercategory>_<sampling_method>_<index>`
- Custom Name: `<supercategory>_<custom_name>`
### Getting Confusing Classes

Identify significant confusing pairs within a subset based on a misclassification threshold and retrieve the top confusing classes.

```python
threshold = 0.2
top_n = 2
confusing_pairs = s.get_confusing_classes(subset_name, threshold=threshold, top_n=top_n)
print(confusing_pairs)
```

**Output:**

```
[
    {"ground_truth": "Tiger", "confusing_class": "Lion", "rate_of_error": 0.25},
    {"ground_truth": "Leopard", "confusing_class": "Tiger", "rate_of_error": 0.30},
    ...
]
```

**Explanation:**

- **Parameters:**
  - `subset_name` (`str`): The name of the subset.
  - `threshold` (`float`, optional): The minimum misclassification rate to consider a pair as confusing. Defaults to `0.2`.
  - `top_n` (`int`, optional): The number of top confusing classes to retrieve per ground truth class. Defaults to `1`. If `None`, retrieves all confusing classes with rates above the threshold.

- **Functionality:**
  - The method scans the misclassification matrix to identify pairs where the misclassification rate exceeds the specified `threshold`.

### Listing Subset Names

Retrieve a list of all existing subset names:

```python
subset_names = s.get_list_of_subset_names()
print("Available subsets:", subset_names)
```

**Output:**

```
Available subsets: ['Mammals_Random_1', 'Mammals_MyMammalsSubset']
```

### Getting Species in a Subset

Get all species included in a specific subset:

```python
species_list = s.get_all_species_in_subset(subset_name)
print("Species in subset:", species_list)
```

**Output:**

```
Species in subset: ['Lion', 'Tiger', 'Elephant', 'Leopard', 'Cheetah']
```

### Getting Species and Directories in a Subset

Retrieve all species in a subset along with their corresponding image directories:

```python
species_dirs = s.get_all_species_in_subset_and_directories(subset_name)
print("Species and directories in subset:", species_dirs)
```

**Output:**

```
Species and directories in subset: {'Lion': 'path/to/images/lion', 'Tiger': 'path/to/images/tiger'}
```

### Running Classification for a Subset

Run a classification task on the subset and generate a misclassification matrix. You can specify the number of samples and optionally provide paths for base and finetuned models:

```python
s.run_classification_for_subset(subset_name, num_samples=3, base_model_path="path/to/base_model")
```

**Note:** The function saves the misclassification matrix in the subset directory.

### Retrieving Misclassification Matrix as a DataFrame

Get the misclassification matrix for the subset in a Pandas DataFrame format:

```python
misclassification_matrix = s.get_misclassification_matrix_pd(subset_name)
print(misclassification_matrix)
```

**Output:**

```
            Lion  Tiger  Elephant  Leopard  Cheetah
Lion        1.0    0.0      0.0      0.0     0.0
Tiger       0.1    0.9      0.0      0.0     0.0
Elephant    0.0    0.0      1.0      0.0     0.0
Leopard     0.0    0.2      0.0      0.8     0.0
Cheetah     0.0    0.0      0.0      0.0     1.0
```

### Retrieving Misclassification Matrix as a Dictionary

Convert the misclassification matrix to a dictionary format, where each ground truth class maps to a dictionary of misclassification scores:

```python
misclassification_dict = s.get_misclassfication_in_dict(subset_name)
print(misclassification_dict)
```

**Output:**

```
{'Lion': {'Lion': 1.0, 'Tiger': 0.0, 'Elephant': 0.0, 'Leopard': 0.0, 'Cheetah': 0.0}, 'Tiger': {'Lion': 0.1, 'Tiger': 0.9, 'Elephant': 0.0, 'Leopard': 0.0, 'Cheetah': 0.0}}
```

### Removing a Subset

Remove a specified subset and all its contents:

```python
s.remove_subset(subset_name)
print(f"Subset '{subset_name}' removed.")
```

**Output:**

```
Subset 'Mammals_Random_1' removed.
```


### Running Classification for a Specific Pair on Validation Set

Perform classification specifically for a pair of species on the validation set:

```python
species_pair = ["Indian Palm Squirrel", "Round-tailed Ground Squirrel"]
misclassification_matrix_val = s.run_classification_for_pair_on_validation(
    name=subset_name,
    pair=species_pair,
    base_model_path="path/to/base_model",
    finetuned_model_path="path/to/finetuned_model",  # Optional
    filtered_images_dir = "/data2/mohant/new_val" # Optional. Change this path if needed
)
print(misclassification_matrix_val)
```

**Output:**

```
   Indian Palm Squirrel  Round-tailed Ground Squirrel
0                   0.95                          0.05
1                   0.10                          0.90
```

### Retrieving Classification for a Pair as a Dictionary

Retrieve the misclassification matrix for a specific pair on the validation set as a nested dictionary:

```python
classification_dict_val = s.get_classification_for_pair_on_validation_as_dict(
    name=subset_name, 
    pair=species_pair,
    base_model_path="path/to/base_model",
    finetuned_model_path="path/to/finetuned_model"  # Optional
)
print(classification_dict_val)
```

**Output:**

```
{
    'Indian Palm Squirrel': {
        'Indian Palm Squirrel': 0.95,
        'Round-tailed Ground Squirrel': 0.05
    },
    'Round-tailed Ground Squirrel': {
        'Indian Palm Squirrel': 0.10,
        'Round-tailed Ground Squirrel': 0.90
    }
}
```

## Example Script

Here's an example script using `SubsetManager`:

```python
from SubsetManager.SubsetManager import SubsetManager

# Initialize SubsetManager
s = SubsetManager()

# Create a subset for the supercategory "Mammals"
name = s.create_subset("Mammals", size=10)
print(f"Created subset: {name}")

# List all subset names
print(s.get_list_of_subset_names())

# Get all species in the created subset
all_species = s.get_all_species_in_subset(name)
print("Species in subset:", all_species)

# Get species and directories
species_dirs = s.get_all_species_in_subset_and_directories(name)
print("Species and directories in subset:", species_dirs)

# Run classification and generate a misclassification matrix
s.run_classification_for_subset(name, num_samples=3)

# Get the misclassification matrix as a Pandas DataFrame
m = s.get_misclassification_matrix_pd(name)
print(m)

# Get the misclassification matrix as a dictionary
misclassification_dict = s.get_misclassfication_in_dict(name)
print(misclassification_dict)

# Remove the subset
s.remove_subset(name)
print(f"Subset '{name}' removed.")
```
