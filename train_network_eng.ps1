# https://github.com/cloneofsimo/lora
# https://github.com/kohya-ss/sd-scripts
# https://rentry.org/2chAI_LoRA_Dreambooth_guide

##### Config start #####

$sd_scripts_dir = "X:\git-repos\sd-scripts\" # Path to kohya-ss/sd-scripts repository 

$ckpt = "X:\SD-models\checkpoint.safetensors" # Path to checkpoint (ckpt / safetensors)
$is_sd_v2_ckpt = 0 # '1' if loading SD 2.x ckeckpoint
$is_sd_v2_768_ckpt = 0 # '1', if loding SD 2.x-768 checkpoint
$image_dir = "X:\training_data\img\" # Path to training images folder
$reg_dir = "X:\training_data\img_reg\" # Path to regularization folder (path can lead to an empty folder, but folder must exist)
$output_dir = "X:\LoRA\" # LoRA network saving path
$output_name = "my_LoRA_network_v1" # LoRA network file name (no extension)

$train_batch_size = 1 # How much images to train simultaneously. Higher number = less training steps (faster), higher VRAM usage
$resolution = 512 # Training resolution
$num_epochs = 10 # Number of epochs
$save_every_n_epochs = 1 # Save every n epochs
$save_last_n_epochs = 999 # Save only last n epochs
$max_token_length = 75 # Max token length. Possible values: 75 / 150 / 225
$clip_skip = 1 # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#ignore-last-layers-of-clip-model

# Advanced settings

$learning_rate = 1e-4 # Learning rate
$unet_lr = $learning_rate # U-Net learning rate
$text_encoder_lr = $learning_rate # Text encoder learning rate
$scheduler = "cosine_with_restarts" # linear, cosine, cosine_with_restarts, polynomial, constant (по умолчанию), constant_with_warmup
$network_dim = 128 # Size of network. Higher number = higher accuracy, higher output file size
$save_precision = "fp16" # None, float, fp16, bf16
$mixed_precision = "fp16" # no, fp16, bf16
$is_random_seed = 1 # 1 = random seed, 0 = static seed
$shuffle_caption = 1 # Shuffle comma-separated captions
$use_vae = 0 
$vae_path = "X:\SD-models\checkpoint.vae.pt"

# Logging

$logging_enabled = 0
$logging_dir = "X:\LoRA\logs\"
$log_prefix = $output_name

##### Config end #####

function Is-Numeric ($value) { return $value -match "^[\d\.]+$" }

function Write-ColorOutput($ForegroundColor)
{
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) { Write-Output $args }
    else { $input | Write-Output }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-Output "Calculating number of images in folders"
$total = 0
$is_structure_wrong = 0
$abort_script = 0
$iter = 0

Get-ChildItem -Path $image_dir -Directory | ForEach-Object {
    $parts = $_.Name.Split("_")
    if (!(Is-Numeric $parts[0]))
    {
		Write-ColorOutput red "Error in $($_):`n`t$($parts[0]) is not a number"
		$is_structure_wrong = 1
        return
    }
	if ([int]$parts[0] -le 0)
	{
		Write-ColorOutput red "Error in $($_):`nNumber of repeats in folder name must be >0"
		$is_structure_wrong = 1
        return
	}
    $repeats = [int]$parts[0]
    $imgs = Get-ChildItem $_.FullName -Depth 0 -File -Include *.jpg, *.png, *.webp | Measure-Object | ForEach-Object { $_.Count }
	if ($iter -eq 0) { Write-Output "Training images:" }
    $img_repeats = ($repeats * $imgs)
    Write-Output "`t$($parts[1]): $repeats repeats * $imgs images = $($img_repeats)"
    $total += $img_repeats
	$iter += 1
}

$iter = 0

if ($is_structure_wrong -eq 0) { Get-ChildItem -Path $reg_dir -Directory | % { if ($abort_script -ne "n") { ForEach-Object {
    $parts = $_.Name.Split("_")
    if (!(Is-Numeric $parts[0]))
    {
		Write-ColorOutput red "Error in $($_):`n`t$($parts[0]) is not a number"
		$is_structure_wrong = 1
        return
    }
	if ([int]$parts[0] -le 0)
	{
		Write-ColorOutput red "Error in $($_):`nNumber of repeats in folder name must be >0"
		$is_structure_wrong = 1
        return
	}
    $repeats = [int]$parts[0]
    $reg_imgs = Get-ChildItem $_.FullName -Depth 0 -File -Include *.jpg, *.png, *.webp | Measure-Object | ForEach-Object { $_.Count }
	if ($iter -eq 0) { Write-Output "Regularization images:" }
	if ($reg_imgs -eq 0)
	{
		Write-ColorOutput darkyellow "Warning: regularization images folder exists, but is empty"
		do { $abort_script = Read-Host "Abort script? (y/N)" }
		until (($abort_script -eq "y") -or ($abort_script -ceq "N"))
		return
	}
	else
	{
		$img_repeats = ($repeats * $reg_imgs)
		Write-Output "`t$($parts[1]): $repeats repeats * $reg_imgs images = $($img_repeats)"
		$iter += 1
	}
} } } }

if ($is_structure_wrong -eq 0 -and ($abort_script -eq "n" -or $abort_script -eq 0))
{
	if ($reg_imgs -gt 0)
	{
		$total *= 2
		Write-Output "Training steps number doubled: number of regularization images >0"
	}
	
	Write-Output "Image number with repeats: $total"
	Write-Output "Training batch size: $train_batch_size"
	Write-Output "Number of epochs: $num_epochs"
	$max_training_steps = [int]($total / $train_batch_size * $num_epochs)
	Write-Output "Number of steps: $total / $train_batch_size * $num_epochs = $max_training_steps"
	
	if ($is_random_seed -le 0) { $seed = 1337 }
	else { $seed = Get-Random }
	
	$image_dir = $image_dir.TrimEnd("\", "/")
	$reg_dir = $reg_dir.TrimEnd("\", "/")
	$output_dir = $output_dir.TrimEnd("\", "/")
	$logging_dir = $logging_dir.TrimEnd("\", "/")
	
	$run_parameters = "--network_module=networks.lora --pretrained_model_name_or_path=`"$ckpt`" --train_data_dir=`"$image_dir`" --reg_data_dir=`"$reg_dir`" --output_dir=`"$output_dir`" --output_name=`"$output_name`" --caption_extension=`".txt`" --resolution=$resolution --prior_loss_weight=1 --enable_bucket --min_bucket_reso=256 --max_bucket_reso=1024 --train_batch_size=$train_batch_size --learning_rate=$learning_rate --unet_lr=$unet_lr --text_encoder_lr=$text_encoder_lr --max_train_steps=$max_training_steps --use_8bit_adam --xformers --save_every_n_epochs=$save_every_n_epochs --save_last_n_epochs=$save_last_n_epochs --save_model_as=safetensors --clip_skip=$clip_skip --seed=$seed --network_dim=$network_dim --cache_latents --lr_scheduler=$scheduler --mixed_precision=$mixed_precision --save_precision=$save_precision"
	
	if ($max_token_length -eq 75) { }
	else
	{
		if ($max_token_length -eq 150 -or $max_token_length -eq 225) { $run_parameters += " --max_token_length=$($max_token_length)" }
		else { Write-ColorOutput darkyellow "The max_token_length is incorrect! Use value 75" }
	}
	
	if ($is_sd_v2_ckpt -le 0) { Write-Output "Stable Diffusion 1.x checkpoint" }
	if ($is_sd_v2_ckpt -ge 1)
	{
		if ($is_sd_v2_768_ckpt -ge 1)
		{
			$v2_resolution = "768"
			$run_parameters += " --v_parameterization"
		}
		else { $v2_resolution = "512" }
		Write-Output "Stable Diffusion 2.x ($v2_resolution) checkpoint"
		$run_parameters += " --v2"
		if ($clip_skip -eq -not 1)
		{
			Write-ColorOutput darkyellow "Warning: training results of SD 2.x checkpoint with clip_skip other than 1 might be unpredictable"
			do { $abort_script = Read-Host "Abort script? (y/N)" }
			until (($abort_script -eq "y") -or ($abort_script -ceq "N"))
		}
	}
	
	if ($shuffle_caption -ge 1) { $run_parameters += " --shuffle_caption" }
	
	if ($logging_enabled -ge 1) { $run_parameters += " --logging_dir=`"$logging_dir`" --log_prefix=`"$output_name`""}
	
	if ($use_vae -ge 1) { $run_parameters += " --vae=`"$vae_path`"" }
	
	sleep -s 1
	
	if ($abort_script -eq "n" -or $abort_script -eq 0)
	{
		Write-ColorOutput green "Running script with parameters:"
		sleep -s 1
		Write-Output "$($run_parameters -split '--' | foreach { if ($_ -ceq '') { Write-Output '' } else { Write-Output --`"$_`n`" } } | foreach { $_ -replace '=', ' = ' })"
		$script_origin = (get-location).path
		cd $sd_scripts_dir
		.\venv\Scripts\activate
		powershell accelerate launch --num_cpu_threads_per_process 12 train_network.py $run_parameters
		deactivate
		cd $script_origin
	}
}

# 13.01.23 by anon