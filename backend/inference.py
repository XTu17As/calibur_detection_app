import os
import json
import math 
import io 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms
from torchvision.ops import nms, box_iou 
from PIL import Image 
import numpy as np 


# Detection configuration
INFERENCE_THRESHOLD = 0.0
NMS_IOU_THRESH = 0.5 
SOFT_NMS_SIGMA = 0.0
SOFT_NMS_MIN_SCORE = 0.1 

# Image and model configuration
MAX_STRIDE = 128
STRIDES = [8, 16, 32, 64, 128]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    Uses global pooling and two fully-connected layers to learn
    channel importance weights.
    """
    def __init__(self, in_channels, rd_ratio=0.25):
        super().__init__()
        rd_channels = int(in_channels * rd_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, rd_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class MBConv(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block (MBConv).
    Expands channels, applies depthwise convolution, adds SE attention,
    and projects back to output channels. Includes residual connection
    when input and output dimensions match.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_channels = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        
        # Expansion phase (only if expansion ratio > 1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.GELU())
        
        # Depthwise convolution phase
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.GELU())
        
        # Squeeze-and-Excitation
        layers.append(SqueezeExcite(hidden_channels))
        
        # Projection phase
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DropPath(nn.Module):
    """
    Stochastic Depth (also known as Drop Path).
    Randomly drops entire residual branches during training
    to improve model regularization.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: 
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = torch.rand(shape, device=x.device) >= self.drop_prob
        return x / (1.0 - self.drop_prob) * keep


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows for window-based attention.
    Reshapes [B, H, W, C] into [B * num_windows, window_size * window_size, C].
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size*window_size, C)
    return x


def window_reverse(windows, window_size, H, W):
    """
    Reverse the window partition operation.
    Reconstructs the original feature map from windowed patches.
    """
    B = int(windows.shape[0] / (H * W / (window_size * window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding layer.
    Converts input images into patch tokens using a convolutional projection.
    """
    def __init__(self, in_ch=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = self.norm(x)
        x = x.flatten(2).transpose(1,2)
        return x, Hp, Wp


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention module.
    Computes attention within local windows rather than globally,
    reducing computational complexity for high-resolution inputs.
    Includes relative position biases for better spatial awareness.
    """
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Create relative position bias table
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords = coords.flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        
        self.register_buffer('relative_index', 
            ((relative_coords[:,:,0] + window_size - 1) * (2*window_size-1) + 
             (relative_coords[:,:,1] + window_size - 1)).flatten())
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.window_size
        x4 = x.view(B, H, W, C)
        
        # Pad if necessary to make dimensions divisible by window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x4 = F.pad(x4, (0,0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = x4.shape[1], x4.shape[2]
        x_windows = window_partition(x4, ws)
        
        # Compute QKV projections
        qkv = self.qkv(x_windows).reshape(x_windows.size(0), x_windows.size(1), 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.unbind(2)
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        
        # Scaled dot-product attention with relative position bias
        attn = (q @ k.transpose(-2,-1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_index].view(ws*ws, ws*ws, -1).permute(2,0,1)
        attn = attn + bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(x_windows.size(0), x_windows.size(1), C)
        out = self.proj(out)
        
        # Reverse window partition and remove padding
        x = window_reverse(out, ws, Hp, Wp)
        x = x[:, :H, :W, :].reshape(B, H*W, C)
        return x


class SwinBlock(nn.Module):
    """
    Swin Transformer Block.
    Combines window-based attention with shifted windows for cross-window
    connections, followed by a feed-forward MLP. Uses pre-normalization
    and residual connections.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), 
            nn.GELU(), 
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x4 = x.view(B, H, W, C)
        
        # Apply cyclic shift for shifted window attention
        if self.shift_size > 0:
            x4 = torch.roll(x4, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        
        x = x4.view(B, H*W, C)
        x = self.attn(x, H, W)
        x = x.view(B, H, W, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        
        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for downsampling.
    Reduces spatial resolution by 2x while increasing channel dimensions.
    Concatenates 2x2 neighboring patches and applies linear projection.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim * 4)
        self.reduction = nn.Linear(in_dim * 4, out_dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x4 = x.view(B, H, W, C)
        
        # Pad if dimensions are odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x4 = F.pad(x4, (0,0,0, pad_w, 0, pad_h))
        
        Hp, Wp = x4.shape[1], x4.shape[2]
        
        # Extract 2x2 patches
        x0 = x4[:, 0::2, 0::2, :]
        x1 = x4[:, 1::2, 0::2, :]
        x2 = x4[:, 0::2, 1::2, :]
        x3 = x4[:, 1::2, 1::2, :]
        
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4*C)
        x_cat = self.norm(x_cat)
        x_out = self.reduction(x_cat)
        return x_out, Hp//2, Wp//2


class TinyViT5MBackbone(nn.Module):
    """
    TinyViT Backbone network.
    A lightweight vision transformer combining convolutional blocks
    in early stages with Swin Transformer blocks in later stages.
    Outputs multi-scale feature maps for object detection.
    """
    def __init__(self, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2], 
                 num_heads=[2, 4, 5, 10], window_size=7, drop_path_rate=0.1, expand_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch=3, embed_dim=embed_dims[0], patch_size=4)
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        
        # Calculate stochastic depth drop rates
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0
        
        # Stage 1: Convolutional blocks
        stage1_blocks = []
        for j in range(depths[0]):
            stage1_blocks.append(MBConv(embed_dims[0], embed_dims[0], stride=1, expand_ratio=expand_ratio))
        self.stages.append(nn.ModuleList(stage1_blocks))
        cur += depths[0]
        self.mergers.append(PatchMerging(embed_dims[0], embed_dims[1]))
        
        # Stages 2-4: Swin Transformer blocks
        for i in range(1, 4):
            blocks = []
            for j in range(depths[i]):
                # Alternate between regular and shifted window attention
                shift = (j % 2) * (window_size // 2)
                blocks.append(SwinBlock(
                    embed_dims[i], input_resolution=(1,1), num_heads=num_heads[i],
                    window_size=window_size, shift_size=shift, drop_path=dp_rates[cur + j]
                ))
            self.stages.append(nn.ModuleList(blocks))
            cur += depths[i]
            if i < 3:
                self.mergers.append(PatchMerging(embed_dims[i], embed_dims[i+1]))
    
    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        C = x.shape[2]
        
        # Process first stage with convolutional blocks
        x_conv = x.transpose(1, 2).contiguous().view(B, C, H, W)
        for blk in self.stages[0]:
            x_conv = blk(x_conv)
        x, H, W = self.mergers[0](x_conv.flatten(2).transpose(1,2), H, W)
        
        # Process remaining stages and collect feature maps
        outs = []
        for i, stage in enumerate(self.stages[1:], start=1):
            for blk in stage:
                x = blk(x, H, W)
            C = x.shape[2]
            fmap = x.transpose(1,2).contiguous().view(B, C, H, W)
            outs.append(fmap)
            if i < len(self.mergers):
                x, H, W = self.mergers[i](x, H, W)
        
        c3, c4, c5 = outs[0], outs[1], outs[2]
        return [c3, c4, c5]


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Builds a multi-scale feature pyramid with top-down pathway
    and lateral connections. Generates five pyramid levels (P3-P7)
    for detecting objects at different scales.
    """
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3,1,1) for _ in in_channels])
        self.extra_p6 = nn.Conv2d(out_channels, out_channels, 3,2,1)
        self.extra_p7 = nn.Conv2d(out_channels, out_channels, 3,2,1)
    
    def forward(self, inputs):
        c3, c4, c5 = inputs
        
        # Build top-down pathway with lateral connections
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        # Apply output convolutions to reduce aliasing
        p3 = self.output_convs[0](p3)
        p4 = self.output_convs[1](p4)
        p5 = self.output_convs[2](p5)
        
        # Add coarser pyramid levels through downsampling
        p6 = self.extra_p6(p5)
        p7 = self.extra_p7(p6)
        
        return [p3, p4, p5, p6, p7]


class FCOSHead(nn.Module):
    """
    FCOS Detection Head for anchor-free object detection.
    Predicts class logits, bounding box regressions (as distances from point),
    and centerness scores at each spatial location.
    """
    def __init__(self, num_classes, in_channels=128):
        super().__init__()
        cls_layers = []
        reg_layers = []
        num_convs = 4
        
        # Build parallel classification and regression towers
        for _ in range(num_convs):
            cls_layers += [nn.Conv2d(in_channels, in_channels, 3,1,1, bias=True), 
                          nn.GroupNorm(32,in_channels), nn.ReLU()]
            reg_layers += [nn.Conv2d(in_channels, in_channels, 3,1,1, bias=True), 
                          nn.GroupNorm(32,in_channels), nn.ReLU()]
        
        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3,1,1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3,1,1)
        self.centerness = nn.Conv2d(in_channels, 1, 3,1,1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize classification bias for better training stability
        pi = 0.01
        bias_value = -math.log((1 - pi) / pi)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
    
    def forward(self, features):
        cls_outs, reg_outs, cent_outs = [], [], []
        for f in features:
            cls_t = self.cls_tower(f)
            reg_t = self.reg_tower(f)
            cls_outs.append(self.cls_logits(cls_t))
            reg_outs.append(F.relu(self.bbox_pred(reg_t)))
            cent_outs.append(self.centerness(reg_t))
        return cls_outs, reg_outs, cent_outs


class TinyViT_FPN_FCOS(nn.Module):
    """
    Complete object detection model.
    Combines TinyViT backbone, FPN neck, and FCOS detection head
    for end-to-end anchor-free object detection.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = TinyViT5MBackbone(
            embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10], window_size=7, drop_path_rate=0.1
        )
        self.fpn = FPN(in_channels=[128, 160, 320], out_channels=128)
        self.head = FCOSHead(num_classes, in_channels=128)
    
    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats)
        cls, reg, cen = self.head(fpn_feats)
        return cls, reg, cen, fpn_feats


def compute_locations(feature, stride, device=None):
    """
    Compute spatial locations (centers) for each position in the feature map.
    Returns tensor of (x, y) coordinates in the original image space.
    """
    if isinstance(feature, torch.Tensor):
        _,_,h,w = feature.shape
    else:
        _,_,h,w = feature
    
    device = device if device is not None else feature.device if isinstance(feature, torch.Tensor) else torch.device(DEVICE)
    shifts_x = (torch.arange(0, w, device=device) + 0.5) * stride
    shifts_y = (torch.arange(0, h, device=device) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return torch.stack((shift_x, shift_y), dim=1)


def decode_boxes(locations, ltrb):
    """
    Decode bounding boxes from FCOS predictions.
    Converts (center_x, center_y) and (left, top, right, bottom) distances
    into (x1, y1, x2, y2) box coordinates.
    """
    x = locations[:,0]
    y = locations[:,1]
    l,t,r,b = ltrb[:,0], ltrb[:,1], ltrb[:,2], ltrb[:,3]
    return torch.stack([x - l, y - t, x + r, y + b], dim=1)


def soft_nms(boxes, scores, labels, sigma=0.5, min_score_thresh=0.001):
    """
    Soft Non-Maximum Suppression.
    Gradually decays detection scores based on IoU overlap rather than
    hard removal. This preserves detections of nearby objects while
    still suppressing duplicates.
    """
    if boxes.numel() == 0: 
        return boxes, scores, labels
    
    order = scores.argsort(descending=True)
    keep_indices = []
    decayed_scores = scores.clone()
    
    while order.numel() > 0:
        i = order[0]
        keep_indices.append(i)
        if order.numel() == 1: 
            break
        
        current_box = boxes[i].unsqueeze(0)
        other_boxes = boxes[order[1:]]
        ious = box_iou(current_box, other_boxes).squeeze(0)
        
        # Apply Gaussian decay based on IoU
        decay_factor = torch.exp(-torch.pow(ious, 2) / sigma)
        decayed_scores[order[1:]] *= decay_factor
        
        # Keep only boxes above threshold
        remaining_mask = decayed_scores[order[1:]] >= min_score_thresh
        order = order[1:][remaining_mask]
        
        # Re-sort by original scores for next iteration
        if order.numel() > 0:
            original_scores_remaining = scores[order]
            new_sort_order = original_scores_remaining.argsort(descending=True)
            order = order[new_sort_order]
    
    keep_indices = torch.tensor(keep_indices, device=boxes.device)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def load_model(model_folder: str):
    """
    Load trained model and class names from a model directory.
    Expects 'best_model.pth' and a COCO-format annotation JSON file.
    Handles class count mismatches by loading non-strictly.
    """
    print(f"Loading model from: {model_folder}")
    pth_path = os.path.join(model_folder, "best_model.pth")
    
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"'best_model.pth' not found in {model_folder}")

    # Find annotation file to extract class names
    ann_file = None
    for f in os.listdir(model_folder):
        if f.endswith('.json') and 'instances' in f:
            ann_file = os.path.join(model_folder, f)
            break
            
    if ann_file is None or not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation JSON missing in {model_folder}")

    # Parse class information from annotation file
    try:
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        if 'categories' not in coco_data or not coco_data['categories']:
             raise ValueError("Annotation file missing categories.")
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        class_names = [cat['name'] for cat in categories]
        num_classes = len(class_names)
        if num_classes == 0:
             raise ValueError("Categories list empty.")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        raise

    model = TinyViT_FPN_FCOS(num_classes).to(DEVICE)

    # Load model weights with error handling for class count mismatches
    try:
        map_location = torch.device(DEVICE)
        checkpoint = torch.load(pth_path, map_location=map_location)
        state_dict = checkpoint.get('model_state', checkpoint)
        cls_logit_weight_key = 'head.cls_logits.weight'
        cls_logit_bias_key = 'head.cls_logits.bias'

        # Check for class count mismatch
        if cls_logit_weight_key in state_dict:
            loaded_num_classes = state_dict[cls_logit_weight_key].shape[0]
            if loaded_num_classes != num_classes:
                print(f"WARNING: Class count mismatch ({loaded_num_classes} vs {num_classes}). Loading non-strictly.")
                state_dict.pop(cls_logit_weight_key, None)
                state_dict.pop(cls_logit_bias_key, None)
                model.load_state_dict(state_dict, strict=False)
            else:
                 model.load_state_dict(state_dict)
        else:
             model.load_state_dict(state_dict, strict=False)

        model.eval()
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        raise

    return model, class_names


def preprocess_image(image_bytes: io.BytesIO):
    """
    Preprocess input image for model inference.
    Pads image to be divisible by MAX_STRIDE and applies
    standard ImageNet normalization.
    """
    image = Image.open(image_bytes).convert('RGB')
    w, h = image.size
    
    # Calculate padded dimensions
    new_h = (h + MAX_STRIDE - 1) // MAX_STRIDE * MAX_STRIDE
    new_w = (w + MAX_STRIDE - 1) // MAX_STRIDE * MAX_STRIDE
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor_unpadded = transform(image).to(DEVICE)
    
    # Add padding
    pad_h = new_h - h
    pad_w = new_w - w
    tensor = F.pad(tensor_unpadded, (0, pad_w, 0, pad_h), "constant", 0)
    tensor = tensor.unsqueeze(0)
    
    return image, tensor, (new_w, new_h)


def run_inference(model, class_names: list, tensor: torch.Tensor, 
                  original_image: Image.Image, nms_method: str = 'soft'):
    """
    Run model inference and post-process detections.
    Performs forward pass, decodes predictions, applies score thresholding,
    and runs NMS per class. Returns list of final detections.
    """
    detections = []
    num_classes = len(class_names)
    input_h, input_w = tensor.shape[-2:]

    with torch.no_grad():
        cls_outs, reg_outs, cen_outs, feats = model(tensor)
        locations = [compute_locations(f, s, device=DEVICE).to(DEVICE) 
                    for f, s in zip(feats, STRIDES)]
        all_boxes, all_scores, all_labels = [], [], []

        # Process each pyramid level
        for lvl, (cls_map, reg_map, cen_map, loc) in enumerate(zip(cls_outs, reg_outs, cen_outs, locations)):
            cls_per = cls_map[0].permute(1,2,0).reshape(-1, num_classes).to(DEVICE)
            reg_per = reg_map[0].permute(1,2,0).reshape(-1,4).to(DEVICE)
            cen_per = torch.sigmoid(cen_map[0].permute(1,2,0).reshape(-1)).to(DEVICE)
            
            scores_per = torch.sigmoid(cls_per)
            max_scores, labels = torch.max(scores_per, 1)
            
            # Combine classification and centerness scores
            final_scores_t = max_scores * cen_per
            keep_mask_t = final_scores_t > INFERENCE_THRESHOLD

            if keep_mask_t.sum().item() == 0: 
                continue

            # Decode boxes and clamp to image bounds
            loc_sel = loc[keep_mask_t]
            reg_sel = reg_per[keep_mask_t]
            labels_sel = labels[keep_mask_t]
            scores_sel = final_scores_t[keep_mask_t]
            boxes_sel = decode_boxes(loc_sel, reg_sel * STRIDES[lvl])
            boxes_sel[:, 0] = boxes_sel[:, 0].clamp(min=0, max=input_w)
            boxes_sel[:, 1] = boxes_sel[:, 1].clamp(min=0, max=input_h)
            boxes_sel[:, 2] = boxes_sel[:, 2].clamp(min=0, max=input_w)
            boxes_sel[:, 3] = boxes_sel[:, 3].clamp(min=0, max=input_h)

            all_boxes.append(boxes_sel)
            all_scores.append(scores_sel)
            all_labels.append(labels_sel)

    if not all_boxes: 
        return []

    pred_boxes_tensor = torch.cat(all_boxes)
    pred_scores_tensor = torch.cat(all_scores)
    pred_labels_tensor = torch.cat(all_labels)

    # Apply NMS per class
    final_boxes_list, final_scores_list, final_labels_list = [], [], []
    unique_labels = pred_labels_tensor.unique()

    for label_id in unique_labels:
        class_mask = (pred_labels_tensor == label_id)
        class_boxes = pred_boxes_tensor[class_mask]
        class_scores = pred_scores_tensor[class_mask]
        if class_boxes.numel() == 0: 
            continue

        if nms_method == 'soft':
            keep_boxes, keep_scores, _ = soft_nms(
                class_boxes, class_scores,
                torch.full_like(class_scores, label_id),
                sigma=SOFT_NMS_SIGMA, min_score_thresh=SOFT_NMS_MIN_SCORE
            )
            final_boxes_list.append(keep_boxes)
            final_scores_list.append(keep_scores)
            final_labels_list.append(torch.full_like(keep_scores, label_id, dtype=torch.long))
        else:
            keep_indices = nms(class_boxes, class_scores, NMS_IOU_THRESH)
            final_boxes_list.append(class_boxes[keep_indices])
            final_scores_list.append(class_scores[keep_indices])
            final_labels_list.append(torch.full_like(class_scores[keep_indices], label_id, dtype=torch.long))

    if not final_boxes_list: 
        return []

    # Format final detections
    final_boxes = torch.cat(final_boxes_list).cpu().numpy()
    final_scores = torch.cat(final_scores_list).cpu().numpy()
    final_labels = torch.cat(final_labels_list).cpu().numpy()

    for i in range(len(final_scores)):
        score = float(final_scores[i])
        label_idx = int(final_labels[i])
        label_name = class_names[label_idx] if label_idx < len(class_names) else "UNKNOWN"
        box = final_boxes[i]
        detections.append({
            "label": label_name,
            "score": score,
            "box": box.tolist()
        })

    return detections
