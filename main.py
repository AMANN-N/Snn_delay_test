
if __name__ == '__main__':
    #from datasets import SHD_dataloaders
    from datasets import ESC_50
    #from datasets import SSC_dataloaders
    from config import Config
    from snn_delays import SnnDelays
    import torch
    from snn import SNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===> Device = {device}")

    config = Config()


    #train_loader, valid_loader= SHD_dataloaders(config) 
    train_loader , val_loader , test_loader  = ESC_50(config)
    #model = SNN(config).to(device)
    model = SnnDelays(config).to(device)

    #model.round_pos()

    model.train_model(train_loader, val_loader, test_loader , device)
