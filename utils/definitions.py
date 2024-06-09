# Paths in remote server
NEUROVOZ_PARENT_REMOTE = r'/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/'
NEUROVOZ_ALIGNED = NEUROVOZ_PARENT_REMOTE + r'neurovoz_htk_forced_alignment/' + r'texts'

ALBAYCIN_PARENT_REMOTE = r'/media/my_ftp/BasesDeDatos_Voz_Habla/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/'
ALBAYCIN_ALIGNED = ALBAYCIN_PARENT_REMOTE + r'albayzin_htk_forced_alignment'


# Local folder to store NeuroVoz labels
NEUROVOZ_LABELS_LOCAL = r'labeled/NeuroVoz'
# Local folder to store already processed data
PROCESSED_DATA_LOCAL = r'local_results/'
# Local folder where data partitions are stored.
PARTITIONS_DATA_LOCAL = PROCESSED_DATA_LOCAL + r'folds/'
