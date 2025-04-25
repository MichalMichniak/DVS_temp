import torch.nn as nn
import torch


class SpikingDense(nn.Module):
    def __init__(self, units, name, X_n=1, outputLayer=False, robustness_params={}, input_dim=None,
                 kernel_regularizer=None, kernel_initializer=None):
        super().__init__()
        self.units = units
        self.B_n = (1 + 0.5) * X_n
        self.outputLayer=outputLayer
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.noise=robustness_params['noise']
        self.time_bits=robustness_params['time_bits']
        self.weight_bits =robustness_params['weight_bits'] 
        self.w_min, self.w_max=-1.0, 1.0
        self.alpha = torch.full((units,), 1, dtype=torch.float64)
        self.input_dim=input_dim
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer
    
    def build(self, input_dim, kernel : torch.Tensor = None):
        # Ensure input_dim is defined properly if not passed.
        if input_dim[-1] is None:
            input_dim = (None, self.input_dim)

        # Create kernel weights and D_i.
        if kernel is not None:
            self.kernel = nn.Parameter(kernel.clone())
        else:
            self.kernel = nn.Parameter(torch.empty(input_dim[-1], self.units))
        self.D_i = nn.Parameter(torch.zeros(self.units))

        # Apply the initializer if provided.
        if self.initializer:
            self.kernel = self.initializer(self.kernel) # tu zmiana TODO

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n, dtype=torch.float64, requires_grad=False)

        # Returning for function signature consistency
        return t_min, t_min + self.B_n
    
    def forward(self, tj):
        """
        Input spiking times `tj`, output spiking times `ti` or membrane potential value for the output layer.
        """
        # Call the custom spiking logic
        output = call_spiking(tj, self.kernel, self.D_i, self.t_min, self.t_max, noise=self.noise)

        # If this is the output layer, perform the special integration logic
        if self.outputLayer:
            # Compute weighted product
            W_mult_x = torch.matmul(self.t_min - tj, self.kernel)
            self.alpha = self.D_i / (self.t_min - self.t_min_prev)
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x

        return output
    

class SpikingConv2D(nn.Module):
    def __init__(self, filters, name, X_n=1, padding='same', kernel_size=(3,3), robustness_params=None):
        super(SpikingConv2D, self).__init__()
        
        if robustness_params is None:
            robustness_params = {}
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev, self.t_min, self.t_max = 0, 0, 1
        self.w_min, self.w_max = -1.0, 1.0
        self.time_bits = robustness_params.get('time_bits', 1)
        self.weight_bits = robustness_params.get('weight_bits', 1) 
        self.noise = robustness_params.get('noise', 0.0)
        
        # Initialize alpha as a tensor of ones
        self.alpha = nn.Parameter(torch.ones(filters, dtype=torch.float64))
        
        # Registering the kernel as a learnable parameter
        self.kernel = nn.Parameter(torch.randn(filters, 1, kernel_size[0], kernel_size[1], dtype=torch.float64))
        
        # Placeholder for batch normalization parameters
        self.BN = nn.Parameter(torch.tensor([0], dtype=torch.float64), requires_grad=False)
        self.BN_before_ReLU = nn.Parameter(torch.tensor([0], dtype=torch.float64), requires_grad=False)
        
        # Parameter for different thresholds
        self.D_i = nn.Parameter(torch.zeros(9, filters, dtype=torch.float64))

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max = t_min + self.B_n
        return t_min, t_min + self.B_n

    def call_spiking(self, tj, W, D_i, t_min, t_max, noise):
        """
        Calculates spiking times from which ReLU functionality can be recovered.
        """
        threshold = t_max - t_min - D_i
        
        # Calculate output spiking time ti
        ti = torch.matmul(tj - t_min, W) + threshold + t_min
        
        # Ensure valid spiking time
        ti = torch.where(ti < t_max, ti, t_max)
        
        # Add noise
        if noise > 0:
            ti += torch.randn_like(ti) * noise
        
        return ti

    def forward(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        # Image size in case of padding='same' or padding='valid'
        padding_size = int(self.padding == 'same') * (self.kernel_size[0] // 2)
        image_same_size = tj.size(2) 
        image_valid_size = image_same_size - self.kernel_size[0] + 1
        
        # Pad input with t_min value, equivalent to 0 in ReLU network
        tj = torch.nn.functional.pad(tj, (0, 0, padding_size, padding_size, padding_size, padding_size), value=self.t_min)
        
        # Extract image patches
        tj = torch.nn.functional.unfold(tj.unsqueeze(0), kernel_size=self.kernel_size, stride=1).permute(0, 2, 1).contiguous()
        
        # Reshape kernel for fully connected behavior
        W = self.kernel.view(-1, self.filters)

        if self.padding == 'valid' or self.BN != 1 or self.BN_before_ReLU == 1: 
            tj = tj.view(-1, W.size(0))
            ti = self.call_spiking(tj, W, self.D_i[0], self.t_min, self.t_max, noise=self.noise)
            if self.padding == 'valid':
                ti = ti.view(-1, image_valid_size, image_valid_size, self.filters)
            else:
                ti = ti.view(-1, image_same_size, image_same_size, self.filters)
        else:
            # Partition the input for different thresholds
            tj_partitioned = [
                tj[:, 1:-1, 1:-1], 
                tj[:, :1, :1], 
                tj[:, :1, 1:-1], 
                tj[:, :1, -1:], 
                tj[:, 1:-1, -1:], 
                tj[:, -1:, -1:], 
                tj[:, -1:, 1:-1], 
                tj[:, -1:, :1], 
                tj[:, 1:-1, :1]
            ]
            ti_partitioned = []

            for i, tj_part in enumerate(tj_partitioned):
                tj_part = tj_part.view(-1, W.size(0))
                ti_part = self.call_spiking(tj_part, W, self.D_i[i], self.t_min, self.t_max, noise=self.noise)
                
                # Reshape partitions
                if i == 0: 
                    ti_part = ti_part.view(-1, image_valid_size, image_valid_size, self.filters)
                if i in [1, 3, 5, 7]: 
                    ti_part = ti_part.view(-1, 1, 1, self.filters)
                if i in [2, 6]: 
                    ti_part = ti_part.view(-1, 1, image_valid_size, self.filters)
                if i in [4, 8]: 
                    ti_part = ti_part.view(-1, image_valid_size, 1, self.filters)
                ti_partitioned.append(ti_part) 

            # Concatenate to create a complete output
            if image_valid_size != 0:
                ti_top_row = torch.cat([ti_partitioned[1], ti_partitioned[2], ti_partitioned[3]], dim=2)
                ti_middle = torch.cat([ti_partitioned[8], ti_partitioned[0], ti_partitioned[4]], dim=2)
                ti_bottom_row = torch.cat([ti_partitioned[7], ti_partitioned[6], ti_partitioned[5]], dim=2)
                ti = torch.cat([ti_top_row, ti_middle, ti_bottom_row], dim=1)         
            else:
                ti_top_row = torch.cat([ti_partitioned[1], ti_partitioned[3]], dim=2)
                ti_bottom_row = torch.cat([ti_partitioned[7], ti_partitioned[5]], dim=2)
                ti = torch.cat([ti_top_row, ti_bottom_row], dim=1)

        return ti


class ModelTmax(nn.Module):
    def __init__(self, layers, optimizer_class=torch.optim.Adam, **kwargs):
        super(ModelTmax, self).__init__(**kwargs)
        self.layers = nn.ModuleList(layers)  # Store layers as a ModuleList
        self.optimizer = optimizer_class(self.parameters())  # Initialize optimizer
        self.loss_fn = nn.CrossEntropyLoss()  # Example loss function; modify as needed
        self.metrics = []  # Can define metrics as needed

    def train_step(self, data):
        x, y_all = data  # Assuming data is a tuple (inputs, targets)
        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        y_pred_all = self(x)  # Model prediction
        loss = self.loss_fn(y_pred_all, y_all)  # Calculate loss

        # Backward pass
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update parameters

        # Dynamic threshold adjustment
        t_min_prev, t_min, k = 0.0, 1.0, 0
        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Check if layer is Conv or Dense
                try:
                    # Assuming layer has attributes t_max and t_min defined
                    t_max = t_min + max(layer.t_max - layer.t_min, 10.0 * (layer.t_max - torch.min(y_pred_all)))
                except IndexError:
                    t_max = 0

                layer.t_min_prev = t_min_prev  # Assign previous t_min
                layer.t_min = t_min  # Assign current t_min
                layer.t_max = t_max  # Assign calculated t_max
                t_min_prev, t_min = t_min, t_max  # Update t_min and t_min_prev

                if k == len(y_pred_all): 
                    break
                k += 1

        # Update metrics here if applicable
        # This is where you would compute your custom metrics and return them
        return {'loss': loss.item()}  # Return loss as part of metrics

    def test_step(self, data):
        x, y_all = data  # Assuming data is a tuple (inputs, targets)
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            y_pred_all = self(x)  # Model prediction
            loss = self.loss_fn(y_pred_all, y_all)  # Calculate loss

        # Update metrics here if applicable
        return {'loss': loss.item()}  # Return loss as part of metrics

    def forward(self, x):
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x


class VGGLikeReLU(nn.Module):
    def __init__(self, layers2D, layers1D, num_classes, kernel_size=3, dropout=0, bn=False,
                 kernel_regularizer=None, kernel_initializer='glorot_uniform'):
        super(VGGLikeReLU, self).__init__()

        self.convs = nn.ModuleList()  # List to store convolutional layers
        in_channels = 1  # Assuming RGB images; modify if needed
        pool_count = 0
        # Create 2D Convolutional layers
        for i, f in enumerate(layers2D):
            if f != 'pool':
                conv_layer = nn.Conv2d(in_channels, f, kernel_size=kernel_size, padding='same')
                if kernel_regularizer is not None:
                    # Apply weight regularization if specified
                    nn.init.kaiming_uniform_(conv_layer.weight, a=0.1)  # For example
                self.convs.append(conv_layer)
                in_channels = f
                self.add_module(f'relu_{i}', nn.ReLU())
                if bn:
                    self.convs.append(nn.BatchNorm2d(f))
                if dropout > 0:
                    self.convs.append(nn.Dropout(dropout))
            else:
                self.convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
                pool_count+=1

        self.flatten = nn.Flatten()

        self.dense_layers = nn.ModuleList()  # List to store dense layers
        prev = in_channels * (self.image_size // (2**(pool_count)))**2
        for j, d in enumerate(layers1D):
            dense_layer = nn.Linear(in_features=prev, out_features=d)
            prev = d
            self.dense_layers.append(dense_layer)
            self.add_module(f'relu_dense_{j}', nn.ReLU())
            if bn:
                self.dense_layers.append(nn.BatchNorm1d(d))
            if dropout > 0:
                self.dense_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(layers1D[-1], num_classes)  # Final output layer

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def create_vgg_model_ReLU(layers2D, kernel_size, layers1D, data, bn, dropout=0, optimizer='adam'):
    model = VGGLikeReLU(layers2D, layers1D, num_classes=data.num_of_classes,
                        kernel_size=kernel_size, dropout=dropout, bn=bn)
    return model

class VGGLikeSNN(nn.Module):
    def __init__(self, layers2D, layers1D, num_classes, kernel_size=3, X_n=1000,
                 robustness_params=None, kernel_regularizer=None):
        super(VGGLikeSNN, self).__init__()
        self.spiking_convs = nn.ModuleList()
        self.image_size = data.input_shape[0]  # Assuming square images

        for i, f in enumerate(layers2D):
            if f != 'pool':
                self.spiking_convs.append(SpikingConv2D(f, kernel_size, X_n, ...))  # Customize with actual parameters
            else:
                self.spiking_convs.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.flatten = nn.Flatten()

        self.spiking_dense_layers = nn.ModuleList()
        for j, d in enumerate(layers1D):
            self.spiking_dense_layers.append(SpikingDense(d, X_n, ...))  # Customize with actual parameters

        self.output_layer = SpikingDense(num_classes, outputLayer=True)

    def forward(self, x):
        for layer in self.spiking_convs:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.spiking_dense_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, X_n=1000,
                         robustness_params={}, kernel_regularizer=None):
    model = VGGLikeSNN(layers2D, layers1D, num_classes=data.num_of_classes,
                       kernel_size=kernel_size, X_n=X_n, robustness_params=robustness_params)
    return model

# Fully Connected Models

class FullyConnectedReLU(nn.Module):
    def __init__(self, layers=2, num_classes=10, optimizer='adam'):
        super(FullyConnectedReLU, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(784, 340))  # Input layer to first hidden layer
        self.layers.append(nn.ReLU())
        for _ in range(layers - 2):
            self.layers.append(nn.Linear(340, 340))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(340, num_classes)  # Output layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def create_fc_model_ReLU(layers=2, optimizer='adam'):
    model = FullyConnectedReLU(layers=layers, num_classes=10)
    return model

class FullyConnectedSNN(nn.Module):
    def __init__(self, layers=2, optimizer='adam', X_n=1000, robustness_params={}):
        super(FullyConnectedSNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SpikingDense(340, X_n, robustness_params=robustness_params))  # First layer
        for _ in range(layers - 2):
            self.layers.append(SpikingDense(340, X_n, robustness_params=robustness_params))
        self.output_layer = SpikingDense(10, outputLayer=True, robustness_params=robustness_params)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def create_fc_model_SNN(layers, optimizer, X_n=1000, robustness_params={}):
    model = FullyConnectedSNN(layers=layers, optimizer=optimizer, X_n=X_n, robustness_params=robustness_params)
    return model



def call_spiking(tj, W, D_i, t_min, t_max, t_max_next, noise):
    """
    Calculates spiking times to recover ReLU-like functionality.
    Assumes tau_c=1 and B_i^(n)=1.
    """
    # Calculate the spiking threshold (Eq. 18)
    threshold = t_max - t_min - D_i
    
    # Calculate output spiking time ti (Eq. 7)
    # TODO:zmiana
    ti = torch.matmul(tj - t_min, W) + threshold + t_min
    
    # Ensure valid spiking time: do not spike for ti >= t_max
    ti = torch.where(ti < t_max, ti, t_max)
    
    # Add noise to the spiking time for noise simulations
    if noise > 0:
        ti = ti + torch.randn_like(ti) * noise
    
    return ti