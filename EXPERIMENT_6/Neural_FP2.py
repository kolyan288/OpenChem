import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import read_smiles_property_file
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import nfp
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor
from nfp.preprocessing.features import get_ring_size

import tensorflow as tf
from tensorflow.keras import layers
SEED = 123


data = pd.read_csv('./benchmark_datasets/tox21/tox21.csv')
data.fillna(0, inplace=True)                       

# Split the data into training, validation, and test sets
valid, test, train = np.split(data.smiles.sample(frac=1., random_state=1), [300, 600])
len(train), len(valid), len(test)

# FEATURIZE INPUT MOLECULES

cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

def atom_featurizer(atom):
    """ Return an string representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    """ Get a similar classification of the bond type.
    Flipped indicates which 'direction' the bond edge is pointing. """
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()


preprocessor = SmilesPreprocessor(atom_features=atom_featurizer, bond_features=bond_featurizer,
                                  explicit_hs=False)

# Initially, the preprocessor has no data on atom types, so we have to loop over the 
# training set once to pre-allocate these mappings


for smiles in train:
    preprocessor(smiles, train=True)
    


# Construct the tf.data pipeline. There's a lot of specifying data types and
# expected shapes for tensorflow to pre-allocate the necessary arrays. But 
# essentially, this is responsible for calling the input constructor, batching 
# together multiple molecules, and padding the resulting molecules so that all
# molecules in the same batch have the same number of atoms (we pad with zeros,
# hence why the atom and bond types above start with 1 as the unknown class)

train_dataset = tf.data.Dataset.from_generator(
    lambda: ((preprocessor(row.smiles, train=False), row['NR-AR'])
             for i, row in data[data.smiles.isin(train)].iterrows()),
    output_signature=(preprocessor.output_signature, tf.TensorSpec((), dtype=tf.float32)))\
    .cache().shuffle(buffer_size=200)\
    .padded_batch(batch_size=32)\
    .prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(
    lambda: ((preprocessor(row.smiles, train=False), row['NR-AR'])
             for i, row in data[data.smiles.isin(valid)].iterrows()),
    output_signature=(preprocessor.output_signature, tf.TensorSpec((), dtype=tf.float32)))\
    .cache()\
    .padded_batch(batch_size=32)\
    .prefetch(tf.data.experimental.AUTOTUNE)

# DEFINE THE MODEL


# Input layers
atom = layers.Input(shape=[None], dtype=tf.int64, name='atom')
bond = layers.Input(shape=[None], dtype=tf.int64, name='bond')
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

NUM_FEATURES = 512  # Controls the size of the model
UNITS = 256
NUM_HEADS = 4
N_ITERS = 6

# Convert from a single integer defining the atom state to a vector
# of weights associated with that class
atom_state = layers.Embedding(preprocessor.atom_classes, NUM_FEATURES,
                              name='atom_embedding', mask_zero=True)(atom)

# Ditto with the bond state
bond_state = layers.Embedding(preprocessor.bond_classes, NUM_FEATURES,
                              name='bond_embedding', mask_zero=True)(bond)

# Here we use our first nfp layer. This is an attention layer that looks at
# the atom and bond states and reduces them to a single, graph-level vector. 
# mum_heads * units has to be the same dimension as the atom / bond dimension
global_state = nfp.GlobalUpdate(units=UNITS, num_heads=NUM_HEADS)([atom_state, bond_state, connectivity])

for _ in range(N_ITERS):  # Do the message passing
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([bond_state, new_bond_state])
   
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([atom_state, new_atom_state])
    
    new_global_state = nfp.GlobalUpdate(units=UNITS, num_heads=NUM_HEADS)([atom_state, bond_state, connectivity, global_state]) 
    global_state = layers.Add()([global_state, new_global_state])

    
# Since the final prediction is a single, molecule-level property (YSI), we 
# reduce the last global state to a single prediction.
# x = layers.Dense(1024, activation = 'relu')(global_state)
# x = layers.Dense(512, activation = 'relu')(x)
ysi_prediction = layers.Dense(1, activation = 'sigmoid')(global_state)

# Construct the tf.keras model
model = tf.keras.Model([atom, bond, connectivity], [ysi_prediction])
with tf.device('/GPU:0'):
    model.compile(metrics='AUC', loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1E-3))

# Fit the model. The first epoch is slower, since it needs to cache
# the preprocessed molecule inputs
with tf.device('/GPU:0'):
    model.fit(train_dataset, validation_data=valid_dataset, epochs=25)

a = 1