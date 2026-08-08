import torch
from torch_geometric.loader import DataLoader
from models.gnn_matcher import ResumeGNN, ContrastiveMatcher

def train_epoch(model, loader, optimizer, matcher):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Forward pass for resume and JD graphs
        resume_emb = model(batch.resume_x, batch.resume_edge_index, batch.resume_batch)
        jd_emb = model(batch.jd_x, batch.jd_edge_index, batch.jd_batch)
        
        # Contrastive loss
        loss = matcher.contrastive_loss(resume_emb, jd_emb, batch.labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

if __name__ == "__main__":
    # Initialize model
    model = ResumeGNN(input_dim=768, hidden_dim=128, output_dim=64)
    matcher = ContrastiveMatcher(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # TODO: Load your dataset of hired/rejected pairs
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    for epoch in range(50):
        # loss = train_epoch(model, train_loader, optimizer, matcher)
        # print(f"Epoch {epoch}: Loss = {loss:.4f}")
        pass
    
    torch.save(model.state_dict(), "gnn_matcher.pt")
