from rnampnn import *

if __name__=='__main__':
    seeding()
    # split_dataset(gen_dataframe())
    data = RNADataModule.from_defaults()
    model = RNAModel()
    
    trainer = get_trainer(name="RDesign", version=1)
    trainer.fit(model, data)