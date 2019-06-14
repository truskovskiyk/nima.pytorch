
def _create_val_data_part(params: TrainParams):
    val_csv_path = os.path.join(params.path_to_save_csv, 'val.csv')
    test_csv_path = os.path.join(params.path_to_save_csv, 'test.csv')

    transform = Transform()
    val_ds = AVADataset(val_csv_path, params.path_to_images, transform.val_transform)
    test_ds = AVADataset(test_csv_path, params.path_to_images, transform.val_transform)

    val_loader = DataLoader(val_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)

    return val_loader, test_loader


def start_check_model(params: ValidateParams):
    val_loader, test_loader = _create_val_data_part(params)
    model = NIMA()
    model.load_state_dict(torch.load(params.path_to_model_weight))
    criterion = EDMLoss()

    model = model.to(device)
    criterion.to(device)

    val_loss = validate(model=model, loader=val_loader, criterion=criterion)
    test_loss = validate(model=model, loader=test_loader, criterion=criterion)
    return val_loss, test_loss
