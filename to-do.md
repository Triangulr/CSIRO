# CSIRO Biomass – Lightning Pipeline Notes

## Step 0 ─ Environment Prep

```
# Install required dependencies
!pip install -q pytorch-lightning torchmetrics timm

IMPORT os, glob, random, typing.Optional
IMPORT numpy, pandas, matplotlib, seaborn, PIL.Image, tqdm
IMPORT torch, torch.nn, torch.utils.data
IMPORT timm + timm.data helpers
IMPORT pytorch_lightning + callbacks + torchmetrics
PRINT library versions + selected device
```
**Reasoning:** Mirrors the Lightning/TIMM stack required for this pipeline so code, transforms, and logging behave the same way.

## Step 1 ─ Load & Inspect Data

```
PATH_DATA = '/kaggle/input/csiro-biomass'
LOAD train.csv into pandas DataFrame (`df`)
TARGET_COLS = every column except ['image_id','Image']

PRINT dataset shape + target column names
PLOT histograms for numeric targets
PLOT pie charts for categorical columns ('State','target_name')
CONVERT 'Sampling_Date' to datetime → add Day_of_Year → compute corr(target, Day_of_Year)
DEFINE show_images(df, n=12) to display samples sorted by target
ANALYSE image dimensions with PIL + seaborn joint scatter/hist plots
```
**Reasoning:** Matches the exploratory analysis required for this pipeline, including target inspection, categorical balance, temporal signal check, and thorough image diagnostics.

## Step 2 ─ Dataset That Splits Each Image in Two

```
CLASS BiomassDataset(Dataset):
    INIT(df, path_img, transforms=None, mode='train')
        STORE df.reset_index(drop=True)
        self._len = len(df) * 2   # two crops per original image

    __len__ returns self._len

    __getitem__(idx):
        row = df.iloc[idx // 2]
        OPEN RGB image via row['image_path']
        SPLIT: idx % 2 == 0 → left half, else right half
        APPLY transforms if provided
        IF mode == 'test': return tensor, image_path
        ELSE: return tensor, torch.tensor(row['target'], float32)
```
**Reasoning:** This doubling-by-halves trick is central to the data pipeline and must be preserved for both train and test flows.

## Step 3 ─ LightningDataModule Configuration

```
CLASS BiomassDataModule(pl.LightningDataModule):
    INIT(data_path, batch_size=32, img_size=(456,456), val_split=0.2)
        TRAIN TRANSFORMS = torchvision.transforms.Compose([
            RandomResizedCrop(img_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(0.1,0.1,0.1,0.1),
            RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1)),
            GaussianBlur(3),
            ToTensor(),
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        VAL/TEST TRANSFORMS = Compose([
            CenterCrop(img_size),
            ToTensor(),
            Normalize(...)
        ])

        READ train.csv, cache cpu worker count

    setup(stage):
        IF stage in (None,'fit'):
            SHUFFLE df with random_state=42
            SPLIT 80/20 into train_df / val_df
        IF stage in (None,'test'):
            GLOB test jpgs → build DataFrame with sample_id + relative image_path

    train_dataloader → BiomassDataset(train_df,...,mode='train')
    val_dataloader   → BiomassDataset(val_df,...,mode='train') using test_transforms
    test_dataloader  → BiomassDataset(test_df,...,mode='test')

# Example usage
data_module = BiomassDataModule(PATH_DATA, batch_size=8, img_size=(528,528))
data_module.setup()
```
**Reasoning:** Captures the exact augmentation recipe, validation split behaviour, and how the dataloaders are instantiated in the Lightning workflow.

## Step 4 ─ Batch Sanity Checks

```
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
PRINT batch counts
VISUALISE a handful of normalised training tensors (denormalise for display)
```
**Reasoning:** Mirrors the intended verification that augmentation + batching behave correctly and targets line up.

## Step 5 ─ LightningModule Definition

```
CLASS BiomassRegressionModel(pl.LightningModule):
    INIT(model_name='tf_efficientnetv2_m', pretrained=True,
         num_targets=1, learning_rate=0.01, loss_weight_smooth_l1=0.5)

        self.backbone = timm.create_model(model_name,
                                          pretrained=pretrained,
                                          num_classes=0,
                                          global_pool='avg')
        self.regression_head = nn.Linear(backbone.num_features, num_targets)

        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.train_mae / val_mae = torchmetrics.MeanAbsoluteError()
        self.train_mse / val_mse = torchmetrics.MeanSquaredError()

    forward(x): features = backbone(x); return regression_head(features)

    _compute_loss(outputs, targets):
        smooth = SmoothL1(outputs.squeeze(), targets.squeeze())
        mse = MSE(outputs.squeeze(), targets.squeeze())
        return w * smooth + (1-w) * mse, smooth, mse

    training_step → log train_loss, train_mae, train_mse (+ component losses)
    validation_step → log val_loss, val_mae, val_mse
    predict_step → return predictions and image_path for inference loop

    configure_optimizers:
        optimizer = torch.optim.SGD(lr=learning_rate, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                      monitor='val_loss')
        return [optimizer], [scheduler]

# Example instantiation:
model = BiomassRegressionModel(model_name='tf_efficientnet_b6', pretrained=False)
```
**Reasoning:** Encodes the exact Lightning module structure, loss blending, metrics, and optimizer/scheduler choices for this solution.

## Step 6 ─ Trainer, Logging, and Saving

```
logger = CSVLogger('logs', name='biomass_regression')
trainer = pl.Trainer(
    max_epochs=20,
    accelerator='auto',
    devices='auto',
    precision='16-mixed',
    log_every_n_steps=5,
    logger=logger
)

trainer.fit(model, data_module)

torch.save(model.state_dict(), 'biomass_regression_model.pth')
```
**Reasoning:** Reproduces the training loop configuration (no explicit checkpoints/early stopping) and manual state dict export.

## Step 7 ─ Metric Visualisation

```
metrics = pandas.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics.set_index('epoch', inplace=True)
melted = metrics.reset_index().melt(id_vars='epoch')

PLOT (with seaborn) three separate charts on log scale:
  - Loss: ['train_loss_step','val_loss_epoch']
  - MAE: ['train_mae_step','val_mae_epoch']
  - MSE: ['train_mse_step','val_mse_epoch']
```
**Reasoning:** Matches the post-training diagnostics needed to interpret convergence behaviour.

## Step 8 ─ Test-Time Inference & Submission

```
test_loader = data_module.test_dataloader()

# Optional visual sanity check of denormalised test crops

model.eval()
all_preds, all_paths = [], []
with torch.no_grad():
    for images, img_path in test_loader:
        images = images.to(model.device)
        outputs = model(images).squeeze()
        all_preds.extend(np.atleast_1d(outputs.cpu().tolist()))
        all_paths.extend(img_path)

predictions_raw = DataFrame({'image_path': all_paths, 'target': all_preds})
SET negative targets to 0 (clip)
prediction_df = predictions_raw.groupby('image_path')['target'].mean().reset_index()

test_csv = pandas.read_csv(os.path.join(PATH_DATA, 'test.csv'))
submission = test_csv.merge(prediction_df, on='image_path', how='left')
submission[['sample_id','target']].to_csv('submission.csv', index=False)
!head submission.csv
```
**Reasoning:** Precisely reflects the two-half averaging, negative-value clamp, and merge-with-test.csv workflow used to build the Kaggle submission file.

---

These notes summarise the PyTorch Lightning EfficientNet pipeline, covering data augmentation choices, Lightning plumbing, loss design, logging, and the inference/export routine.