import torch
import argparse
from pathlib import Path
import torch.utils.tensorboard as tb
from datetime import datetime
from files.models import load_model, save_model
from files.datasets.road_dataset import load_data
from .metrics import PlannerMetric

def train(
        model_name="transformer_planner",
        num_epochs=50,
        lr=1e-3,
        batch_size=128,
        num_workers=2,
        seed=0,
        exp_dir="logs",
        **kwargs
):
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # set seed
    torch.manual_seed(seed)

    # set logger
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # load model and set model in training mode
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load data
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_data = load_data("drive_data/val", shuffle=False)

    # create optimizer and loss function
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    
    # metrics
    train_planner_metric = PlannerMetric()
    val_planner_metric = PlannerMetric()


    for epoch in range(num_epochs):

        model.train()
        for batch in train_data:
            image = batch["image"].to(device)
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            preds = model(track_left, track_right)

            #mask = waypoints_mask.unsqueeze(-1)
            #loss = loss_func(preds[waypoints_mask], waypoints[waypoints_mask])
            loss = loss_func(preds, waypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_planner_metric.add(preds, waypoints, waypoints_mask)

            global_step += 1

        with torch.inference_mode():

            model.eval()
            for batch in val_data:
                image = batch["image"].to(device)
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
                
                preds = model(track_left, track_right)
                val_planner_metric.add(preds, waypoints, waypoints_mask)

        train_metrics = train_planner_metric.compute()
        val_metrics = val_planner_metric.compute()

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument("--exp_dir", type=str, default="logs")
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--num_epoch", type=int, default=50)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--seed", type=int, default=2024)

        # optional: additional model hyperparamters
        # parser.add_argument("--num_layers", type=int, default=3)

        # pass all arguments to train
        train(**vars(parser.parse_args()))