import numpy as np
import torch

def custom_loss(Y, X_prime, S, time_length):
    X_prime_flat = X_prime.view(-1)  # X'をフラット化
    S = S.reshape(time_length, -1).float()
    SX_prime = torch.matmul(S, X_prime_flat)
    SX_prime = SX_prime / torch.max(SX_prime)
    loss = torch.mean((Y - SX_prime) ** 2)
    return loss

def l1_custom_loss(Y, X_prime, S, time_length, lambda_reg=0.01):
    X_prime_flat = X_prime.view(-1)  # X'をフラット化
    S = S.reshape(time_length, -1).float()
    SX_prime = torch.matmul(S, X_prime_flat)
    SX_prime = SX_prime / torch.max(SX_prime)
    mse_loss = torch.mean((Y - SX_prime) ** 2)
    
    # L1正則化項の計算
    l1_reg = lambda_reg * torch.norm(X_prime_flat, 1)
    
    # 総損失
    loss = mse_loss + l1_reg
    return loss

def tv_custom_loss(Y, X_prime, S, time_length, lambda_reg=0.01):
    X_prime_flat = X_prime.view(-1)  # X'をフラット化
    S = S.reshape(time_length, -1).float()
    SX_prime = torch.matmul(S, X_prime_flat)
    SX_prime = SX_prime / torch.max(SX_prime)
    mse_loss = torch.mean((Y - SX_prime) ** 2)
    
    # Total Variation正則化項の計算
    tv_reg = lambda_reg * torch.sum(torch.abs(X_prime_flat[1:] - X_prime_flat[:-1]))
    
    # 総損失
    loss = mse_loss + tv_reg
    return loss

def l1_tv_custom_loss(Y, X_prime, S, time_length, alpha=0.01, beta=0.01):
    X_prime_flat = X_prime.view(-1)  # X'をフラット化
    S = S.reshape(time_length, -1).float()
    SX_prime = torch.matmul(S, X_prime_flat)
    SX_prime = SX_prime / torch.max(SX_prime)
    mse_loss = torch.mean((Y - SX_prime) ** 2)
    l1_reg = alpha * torch.norm(X_prime_flat, 1)
    tv_reg = beta * torch.sum(torch.abs(X_prime_flat[1:] - X_prime_flat[:-1]))
    loss = mse_loss + l1_reg + tv_reg
    return loss