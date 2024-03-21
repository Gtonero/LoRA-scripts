param([int]$ChainedRun = 0, [int]$TestRun = 0)

<# ##### Config start ##### #>
# Path variables
$name = "name"
$prefix = ""
$sd_scripts_dir = "Folder SD-script" # Path to kohya-ss/sd-scripts repository 
$ckpt = "Model" # Path to checkpoint (ckpt / safetensors)
$is_sd_v2_ckpt = 0 # '1' if loading SD 2.x ckeckpoint
$is_sd_v2_768_ckpt = 0 # '1', if loading SD 2.x-768 checkpoint
$image_dir = "training_image_folder\$name" # Path to training images folder. N_ConceptName inside
$output_dir = "Output\$name$prefix" # LoRA network saving path
$output_name = "$name"+"$prefix" # LoRA network file name (no extension)
$training_comment = ""

# (optional) Additional paths
$enable_netowrk = 0
$network_weights = ""
$dim_from_weights = 1
$reg_dir = "" # Path to regularization folder [reg files can reduce LoRA style effects, but are not recommended for LoRA style use]
$vae_path = "" # Path to VAE [Not recommended if you already use VAE in Webui]

# Main variables
$max_train_epochs = 5     # Number of epochs. Have no power if $desired_training_time > 0
$max_train_steps = 0      # (optional) Custom training steps number. desired_training_time and max_train_epochs must be equal zero for this variable to work
$train_batch_size = 1     # How much images to train simultaneously
                          # Higher number = less training steps (faster), higher VRAM usage
$resolution = "512"       # Training resolution (px)
$save_every_n_epochs = 1  # Save every n epochs
$save_last_n_epochs = 999 # Save only last n epochs
$save_every_n_steps = 0   # Save every n steps
$save_last_n_steps = 0

#State 
$save_state = 0  #need storage 
$resume = ""

$max_token_length = 225 # Max token length. Possible values: 75 / 150 / 225
$clip_skip = 2 # Use output of text encoder from the end of N-th layer
$train_with = 2 # 2=both 1=unet 0=TE

# (optional) Custom training time
$desired_training_time = 0 # If greater than 0, ignore number of images with repetitions when calculating training steps and train network for N minutes
$gpu_training_speed = "" # Average training speed, depending on GPU. Possible values are XX.XXit/s or XX.XXs/it

# Advanced variables
$network_module = "networks.lora" #lycoris.kohya or networks.lora networks.dylora networks.lora_fa
$network_args = "conv_dim=8 conv_alpha=4" #LoRA Conv Trainging slower 10-15%
$enable_args = 0

$optimizer_type = "AdamW" #lion AdaFactor AdamW AdamW8bit Lion8bit prodigy DAdaptAdam DAdaptAdamPreprint Prodigy
#Preset optimizer
$optimizer_args_DAdaption = "decouple=True weight_decay=1 betas=0.9,0.999" 
$optimizer_args_lion = "weight_decay=0.4 betas=0.9,0.99" 
$optimizer_args_adam = "weight_decay=0.1 betas=0.9,0.99"
$optimizer_args_AdaFactor = "relative_step=True scale_parameter=True warmup_init=True"
$optimizer_args_Prodigy = "decouple=True weight_decay=0.01 use_bias_correction=True safeguard_warmup=True betas=0.9,0.99 d_coef=1.5 eps=1e-8 d0=5e-5"

$lr_scheduler_type = ""
$lr_scheduler_args = ""         #Read documentation in python scheduler
$learning_rate = 1e-4           # Learning rate 
$unet_lr = 1.5e-4               # U-Net learning rate
$text_encoder_lr = 0.5          # Text encoder learning rate ratio
$scheduler = "cosine"           # Scheduler to use for learning rate. Possible values: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup
$lr_scheduler_num_cycles = 2    #cosine restart [number] or polynomial power
$lr_warmup_ratio = 0.10         # Ratio of warmup steps in the learning rate scheduler to total training steps (0 to 1)
$network_dim = "16"             # Size (rank) of network. Higher number = higher accuracy, output file size and VRAM usage
$network_alpha = "16"           # Default value - 1
				                # alpha/dim = scale | closer scale is to 0, the greater the diversity of the model. It is recommended to add Unet to compensate for the lower scale

$network_dropout = 0  #recommended 0.1 to 0.3 not recommended >0.5 | 0 = disable
#technique to stop overfitting or reduce overtraining
$scale_weight_norms = 0 #recommend = 1.0 [slower 30% when enable] | 0 = disable | higher low effect
#Prevents overfitting and improves LoRA stability, making it more compatible with other LoRAs

#Validation_dataset [Not working pull request only]
$validation_split = 0          #splits image dataset %
$validation_seed = 1234        #seed
$validation_every_n_step = 0   #check every steps [If this config value is low it will take more time]
$max_validation_steps = 0      #steps check If there are many images, we recommend increasing this value.

#Image Noise //Please choose to use any one
$min_snr_gamma = 0               # value 1-20 recommendation 5 if 0 = disable [if avg loss higher 0.15 set = 5]            
$ip_noise_gamma = 0.1            # idk effects
$debiased_estimation_loss = 0    # idk effects
#Latent Noise
$noise_offset = 0              # recommend <0.1 | 0 = disable If this noise offset is high It will make it similar to using high CFG value
$multires_noise_discount = 0   #recommend 0.1-0.3 | not recommended for LoRA = 0.8 | need iterations enable
$multires_noise_iterations = 0 # 6 to 10 | disable when iterations = 0
#Model supported zero_terminal_snr only 
$zero_terminal_snr = 0  # idk effects 

$bucket_no_upscale = 1  # If the aspect ratio is very different, it is recommended set 1 
                        # maximum image size will be resolution ex.512x512 = 262144 pixel
$bucket_reso_steps = 64 # Use a value that is divisible by 8
                        # less you use this value, the more vram use but makes objects at the edge of the image more unclipped 
                        # Warning maybe unet something worng
$min_bucket_reso = 256 
$max_bucket_reso = 2048 

$is_random_seed = 0        # Seed for training. 1 = random seed, 0 = static seed 1337
$shuffle_caption = 1       # Shuffle comma-separated captions
$keep_tokens = 2           # Keep heading N tokens when shuffling caption tokens
 
#Cache_latents
$random_crop = 0           #Suitable for subjects at the edge of the image. Enabling this will auto disable cache latents
$cache_latents = 1         #It takes longer to prepare the data but will be faster while training
$cache_latents_to_disk = 1 #Suitable for a large number of images The cache will be stored in the training folder [advisable to enable it]
$no_metadata = 0

$min_timestep = 0              # <not recommend> 0 = defualt
$max_timestep = 0              # <not recommend> 0 = defualt

# Script chain running
# Here you specify the paths where the scripts for sequential execution are located
# There could be any number of paths (don't forget to remove < and >)
$script_paths = @(
	"<X:\Path\to\script\script.ps1>",
	"<.\script.ps1>",
	"<script.ps1>"
)

# Additional settings
<# $device = "cuda" #>         # What device to use for training. Posiible values: cuda, cpu
$gradient_checkpointing = 0    # https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing
                               # reduce vram but slow training
$gradient_accumulation_steps = 1 # https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation
$max_data_loader_n_workers = 0   # Max number of CPU threads for DataLoader
                               # The lower the number, the less is RAM consumptiong, faster epoch start and slower data loading
                               # Lower numbers can negatively affect training speed
$save_precision = "bf16"       # Whether to use custom precision for saving, and its type. Possible values: no, float, fp16, bf16
$mixed_precision = "bf16"      # Whether to use mixed precision for training, and its type. Possible values: no, fp16, bf16
$do_not_interrupt = 0 # Do not interrupt script on questionable moments. Enabled by default if running in a chain
$logging_dir = "logging folder" # (optional)
$log_prefix = "$output_name" + "_"
$debug_dataset = 0

# Other settings
$test_run = 0 # Do not launch main script
$do_not_clear_host = 1 # Don't clear console on launch
$dont_draw_flags = 1 # Do not render flags
<# ##### Config end #####  #>

[console]::OutputEncoding = [text.encoding]::UTF8
$current_version = ""
if ($do_not_clear_host -le 0) { Clear-Host } 

function Is-Numeric ($value) { return $value -match "^[\d\.]+$" }
function WCO($BackgroundColor, $ForegroundColor, $NewLine) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args)
	{
		if ($NewLine -eq 1) { Write-Host $args -BackgroundColor $BackgroundColor -ForegroundColor $ForegroundColor -NoNewLine }
		else { Write-Output $args }
	}
    else { $input | Write-Output } 
	$host.UI.RawUI.ForegroundColor = $fc
}
function Get-Changelog {
	$changelog_link = "https://raw.githubusercontent.com/anon-1337/LoRA-scripts/main/english/script_changelog.txt"
	$changelog = (Invoke-WebRequest -Uri $changelog_link).Content | Out-String
	$changelog = $changelog -split "\r?\n"
	$max_version = "0.0"
	$last_version_string_index = 0; $index = 0
	foreach ($line in $changelog) {
		if ($line -match "`#+ v\d+[\.,]\d+") { $max_version = [float]($line -replace "^#+ +v"); $last_version_string_index = $index }
		$index += 1
	}
	$max_version_date = $changelog[$last_version_string_index + 1] -replace "#+ +"
	if ($max_version -gt $current_version) {
		Write-Output ""
		Write-Output "Full changelog:"
		WCO black blue 0 "$changelog_link `n"
		Write-Output "Changes in v${max_version} from ${max_version_date}:"
		while (($last_version_string_index + 3) -le $changelog.Length) { Write-Output "$($changelog[$last_version_string_index + 2])"; $last_version_string_index += 1 }
	}
}

# Autism case #1
if ($dont_draw_flags -le 0) {
$strl = 0
$version_string = "RetardScript $current_version"
$version_string_length = $version_string.Length
while ($strl -lt ($([system.console]::BufferWidth))) { $strl += 1; WCO white white 1 " " }; Write-Output ""; $strl = 0; while ($version_string_length -lt $(($([system.console]::BufferWidth) + $version_string.Length) / 2)) { WCO darkred white 1 " "; $version_string_length += 1 }; WCO darkred white 1 $version_string; $version_string_length = $version_string.Length; while ($version_string_length -lt $(($([system.console]::BufferWidth) + $version_string.Length) / 2 - $version_string.Length % 2 + $([system.console]::BufferWidth) % 2)) { WCO darkred white 1 " "; $version_string_length += 1 }; while ($strl -lt ($([system.console]::BufferWidth))) { $strl += 1; WCO white white 1 " " } }

Write-Output " "
Write-Output "If something not werks or not werks correctly, leave a message here:"
WCO black blue 0 "https://github.com/anon-1337/LoRA-scripts/issues"
Write-Output " "


if ($restart -ne 1) {

$total = 0
$is_structure_wrong = 0
$abort_script = 0
$iter = 0
}
# paths check
$continueLoop = $true
Write-Output "Checking paths..."

$images = Get-ChildItem -Path $image_dir -File -Recurse -Include *.jpg, *.png, *.webp
$imageCount = $images.Count
Write-Host "Total Image Files: $imageCount"

if ($cache_latents -ne 0) { 
$npzFiles = Get-ChildItem -Path $image_dir -Recurse -Filter *.npz
$numberOfNpzFiles = $npzFiles.Count
Write-Host "Total Npz Files: $numberOfNpzFiles" 
}

if ($imageCount -ne $numberOfNpzFiles -and $cache_latents -eq 1) {
    Write-Host "Npz is not equal to Image being checked."
    # Find image 
    $jpgFiles = Get-ChildItem -Path $image_dir -Filter *.jpg -Recurse
    $pngFiles = Get-ChildItem -Path $image_dir -Filter *.png -Recurse
    $webpFiles = Get-ChildItem -Path $image_dir -Filter *.webp -Recurse

    # Find Npz
    $npzFiles = Get-ChildItem -Path $image_dir -Filter *.npz -Recurse

    # Dictionary 
    $missingFiles = @{}

    # check npz subfolder
    foreach ($jpgFile in $jpgFiles) {
        $folder = $jpgFile.Directory.Name
        $imageName = $jpgFile.BaseName
        $npzExist = $npzFiles | Where-Object { $_.Name -eq "$imageName.npz" }
        if (-not $npzExist) {
            if (-not $missingFiles.ContainsKey($folder)) {
                $missingFiles[$folder] = 1
            } else {
                $missingFiles[$folder]++
            }
        }
    }

    foreach ($pngFile in $pngFiles) {
        $folder = $pngFile.Directory.Name
        $imageName = $pngFile.BaseName
        $npzExist = $npzFiles | Where-Object { $_.Name -eq "$imageName.npz" }
        if (-not $npzExist) {
            if (-not $missingFiles.ContainsKey($folder)) {
                $missingFiles[$folder] = 1
            } else {
                $missingFiles[$folder]++
            }
        }
    }
        foreach ($webpFile in $webpFiles) {
        $folder = $pngFile.Directory.Name
        $imageName = $pngFile.BaseName
        $npzExist = $npzFiles | Where-Object { $_.Name -eq "$imageName.npz" }
        if (-not $npzExist) {
            if (-not $missingFiles.ContainsKey($folder)) {
                $missingFiles[$folder] = 1
            } else {
                $missingFiles[$folder]++
            }
        }
    }

    foreach ($folder in $missingFiles.Keys) {
        "`tNo .npz file in $folder = $($missingFiles[$folder])"
    }
}






$all_paths = @( $sd_scripts_dir, $ckpt, $image_dir )
if ($reg_dir -ne "") { $all_paths += $reg_dir }
if ($use_vae -ge 1) { $all_paths += $vae_path }
foreach ($path in $all_paths) {
	if ($path -ne "" -and !(Test-Path $path)) {
		$is_structure_wrong = 1
		Write-Output "Path $path does not exist" } }

if ($is_chained_run -ge 1) { $do_not_interrupt = 1 }

# images
if ($is_structure_wrong -eq 0) { Get-ChildItem -Path $image_dir -Directory | ForEach-Object {
	if ($iter -eq 0) { Write-Output "Calculating number of images in folders" }

    $parts = $_.Name.Split("_")
    if (!(Is-Numeric $parts[0]))
    {
		WCO black red 0 "Error in $($_):`n`t$($parts[0]) is not a number"
		$is_structure_wrong = 1
        return
    }
	if ([int]$parts[0] -le 0)
	{
		WCO black red 0 "Error in $($_):`nNumber of repeats in folder name must be >0"
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
} }

$iter = 0

# regs
if ($is_structure_wrong -eq 0 -and $reg_dir -ne "") { Get-ChildItem -Path $reg_dir -Directory | % { if ($abort_script -ne "y") { ForEach-Object {
    $parts = $_.Name.Split("_")
    if (!(Is-Numeric $parts[0]))
    {
		WCO black red 0 "Error in $($_):`n`t$($parts[0]) is not a number"
		$is_structure_wrong = 1
        return
    }
	if ([int]$parts[0] -le 0)
	{
		WCO black red 0 "Error in $($_):`nNumber of repeats in folder name must be >0"
		$is_structure_wrong = 1
        return
	}
    $repeats = [int]$parts[0]
    $reg_imgs = Get-ChildItem $_.FullName -Depth 0 -File -Include *.jpg, *.png, *.webp | Measure-Object | ForEach-Object { $_.Count }
	if ($iter -eq 0) { Write-Output "Regularization images:" }
	if ($reg_imgs -eq 0 -and $do_not_interrupt -le 0) {
		WCO black darkyellow 0 "Warning: regularization images folder exists, but is empty"
		do { $abort_script = Read-Host "Abort script? (y/N)" }
		until ($abort_script -eq "y" -or $abort_script -ceq "N")
		return }
	else {
		$img_repeats = ($repeats * $reg_imgs)
		Write-Output "`t$($parts[1]): $repeats repeats * $reg_imgs images = $($img_repeats)"
		$iter += 1 }


} } } }

if ($is_structure_wrong -eq 0 -and $abort_script -ne "y")
{
	Write-Output "Training images number with repeats: $total"
	
	# steps
	if ($desired_training_time -gt 0) {
		Write-Output "desired_training_time > 0"
		Write-Output "Using desired_training_time for calculation of training steps, considering the speed of the GPU"
		if ($gpu_training_speed -match '^(?:\d+\.\d+|\d+|\.\d+)(?:(?:it|s)(?:\\|\/)(?:it|s))')
		{
			$speed_value = $gpu_training_speed -replace '[^.0-9]'
			if ([regex]::split($gpu_training_speed, '[\/\\]') -replace '\d+.\d+' -eq 's') { $speed_value = 1 / $speed_value }
			$max_train_steps = [float]$speed_value * 60 * $desired_training_time
			if ($reg_imgs -gt 0) {
				$max_train_steps *= 2
				$max_train_steps = [int]([math]::Round($max_train_steps))
				Write-Output "Number of regularization images greater than 0"
				if ($do_not_interrupt -le 0) { do { $reg_img_compensate_time = Read-Host "Would you like to halve the number of training steps to make up for the increased time? (y/N)" }
				until ($reg_img_compensate_time -eq "y" -or $reg_img_compensate_time -ceq "N") }
				if ($reg_img_compensate_time -eq "y" -or $do_not_interrupt -ge 1) {
					$max_train_steps = [int]([math]::Round($max_train_steps / 2))
					WCO black gray 1 "Total training steps: $([math]::Round($($speed_value * 60), 2)) it/min * $desired_training_time minute(-s) ≈"; WCO white black 1 "$max_train_steps training step(-s)`n" }
				else {
					Write-Output "Your choice is no. Increased time will not be compensated, duration of training is doubled"
					WCO black gray 1 "Total training steps: $([math]::Round($($speed_value * 60), 2)) it/min * $desired_training_time minute(-s) * 2 ≈"; WCO white black 1 "$max_train_steps training step(-s)`n" }
			}
			else {
				$max_train_steps = [int]([math]::Round($max_train_steps))
				WCO black gray 1 "Total training steps: $([math]::Round($($speed_value * 60), 2)) it/min * $desired_training_time minute(-s) ≈"; WCO white black 1 "$max_train_steps training step(-s)`n" }
		}
		else {
			WCO black red 0 "The learning rate is incorrect in gpu_training_speed variable!"
			$abort_script = "y" }
	}
	elseif ($max_train_epochs -ge 1) {
		Write-Output "Using number of training images to calculate total training steps"
		Write-Output "Number of epochs: $max_train_epochs"
		Write-Output "Training batch size: $train_batch_size"
        Write-Output "Training gradient_accumulation_steps: $gradient_accumulation_steps"
		if ($reg_imgs -gt 0)
		{
			$total *= 2
			Write-Output "Number of regularization images is greater than 0: total train steps doubled"
		}
		$max_train_steps = [int]($total / $train_batch_size / $gradient_accumulation_steps * $max_train_epochs)
		WCO black white 1 "Total training steps: $total / $train_batch_size / $gradient_accumulation_steps * $max_train_epochs  =" "$max_train_steps`n"
	}
	else {
		Write-Output "Using custom training steps number"
		WCO black white 1 "Total training steps: "; WCO black green 1 "$max_train_steps`n"
	}

    #check ratio
    $divisionResult = $total / $imageCount
    $divisionResultFormatted = [math]::Round($divisionResult, 2).ToString()
    WCO black white 0 "Image Ratio step = $divisionResultFormatted"

    }



	# run parameters
    
	$run_parameters = "--network_module=`"$network_module`"  --train_data_dir=`"$image_dir`" --highvram " 
    if ( $network_weights -ne "" -and $enable_netowrk -eq "1" ) { $run_parameters += "--network_weights=`"$network_weights`" " }
    if ( $dim_from_weights -eq 1 -and $network_weights -ne "" -and $enable_netowrk -eq "1" ) { $run_parameters += " --dim_from_weight" }
    if ( $validation_split -gt 0 ) { $run_parameters += " --validation_split=$validation_split --validation_seed=$validation_seed" } 
    if ( $validation_every_n_step -gt 0 -and $validation_split -gt 0 ) { $run_parameters += " --validation_every_n_step=$validation_every_n_step" }
    if ( $max_validation_steps -gt 0 -and $validation_split -gt 0 ) { $run_parameters += " --max_validation_steps=$max_validation_steps" } 
	
	# paths
    if ($enable_args -eq 1) { $run_parameters += " --network_args $network_args" }
	$image_dir = $image_dir.TrimEnd("\", "/")
	$reg_dir = $reg_dir.TrimEnd("\", "/")
	$output_dir = $output_dir.TrimEnd("\", "/")
	$logging_dir = $logging_dir.TrimEnd("\", "/")
	if ($reg_dir -ne "") { $run_parameters += " --reg_data_dir=`"$reg_dir`"" }
	$run_parameters += " --output_dir=`"$output_dir`" --output_name=`"$output_name`" --pretrained_model_name_or_path=`"$ckpt`""
	if ($is_sd_v2_ckpt -le 0) { "Stable Diffusion 1.x checkpoint" }
	if ($is_sd_v2_ckpt -ge 1) {
		if ($is_sd_v2_768_ckpt -ge 1) {
			$v2_resolution = "768"
			$run_parameters += " --v_parameterization"
		}
		else { $v2_resolution = "512" }
		Write-Output "Stable Diffusion 2.x ($v2_resolution) checkpoint"
		$run_parameters += " --v2"
		if ($clip_skip -eq -not 1 -and $do_not_interrupt -le 0) {
			WCO black darkyellow 0 "Warning: training results of SD 2.x checkpoint with clip_skip other than 1 might be unpredictable"
			do { $abort_script = Read-Host "Abort script? (y/N)" }
			until ($abort_script -eq "y" -or $abort_script -ceq "N")
		}
	}
	if ($vae_path -ne "") { $run_parameters += " --vae=`"$vae_path`" " }
	
	# main
	if ($desired_training_time -gt 0) { $run_parameters += " --max_train_steps=$([int]$max_train_steps)" }
	elseif ($max_train_epochs -ge 1) { $run_parameters += " --max_train_epochs=$max_train_epochs" }
	else { $run_parameters += " --max_train_steps=$max_train_steps" }
	$run_parameters += " --train_batch_size=$train_batch_size --resolution=$resolution --save_every_n_epochs=$save_every_n_epochs --save_last_n_epochs=$save_last_n_epochs"
    if ($save_every_n_steps -gt 1) { $run_parameters += " --save_every_n_steps=$save_every_n_steps"}
    if ($resume -ne "") { $run_parameters += " --resume=$resume"}
	if ($max_token_length -eq 75) { }
	else {
		if ($max_token_length -eq 150 -or $max_token_length -eq 225) { $run_parameters += " --max_token_length=$($max_token_length)" }
		else { WCO black darkyellow 0 "max_token_length is incorrect! Using value 75" } }
	$run_parameters += " --clip_skip=$clip_skip"
	
	# advanced
    if ($optimizer_type -ne "") { $run_parameters += " --optimizer_type=$optimizer_type" }
    if ($optimizer_type -ieq "lion" -or $optimizer_type -ieq "lion8bit") { $run_parameters += " --optimizer_args $optimizer_args_lion" }
    if ($optimizer_type -ieq "AdamW" -or $optimizer_type -ieq "AdamW8bit") { $run_parameters += " --optimizer_args $optimizer_args_adam" }
    if ($optimizer_type -ieq "DAdaptAdam" -or $optimizer_type -ieq "DAdaptation" ) { $run_parameters += " --optimizer_args $optimizer_args_DAdaption" }
    if ($optimizer_type -ieq "AdaFactor" ) { $run_parameters += " --optimizer_args $optimizer_args_AdaFactor" }
    if ($optimizer_type -ieq "Prodigy" ) { $run_parameters += " --optimizer_args $optimizer_args_Prodigy" }
    if ($training_comment -ne "") { $run_parameters += " --training_comment $training_comment" }
    if ($lr_scheduler_type -ne "" -and $scheduler -eq "") { $run_parameters += " --lr_scheduler_type $lr_scheduler_type" }
    if ($scheduler -eq "cosine_with_restarts" -and $lr_scheduler_type -eq "" ) { $run_parameters += " --lr_scheduler_num_cycle=$lr_scheduler_num_cycles" }
    if ($scheduler -eq "polynomial" -and $lr_scheduler_type -eq "" ) { $run_parameters += " --lr_scheduler_power=$lr_scheduler_num_cycles" }
    if ($lr_scheduler_args -ne "") { $run_parameters += " --lr_scheduler_args $lr_scheduler_args" }
	$run_parameters += " --learning_rate=$learning_rate" 
	if ($unet_lr -ne $learning_rate) { $run_parameters += " --unet_lr=$unet_lr" }
	if ($text_encoder_lr -ne $learning_rate) { 
        $text_encoder_lr_output = $unet_lr * $text_encoder_lr
        $run_parameters += " --text_encoder_lr=$text_encoder_lr_output"
         }
    if ($scheduler -ne "") { $run_parameters += " --lr_scheduler=$scheduler" }
	if ($scheduler -ne "constant" -and $lr_scheduler_type -eq "") {
		if ($lr_warmup_ratio -lt 0.0) { $lr_warmup_ratio = 0.0 }
		if ($lr_warmup_ratio -gt 1.0) { $lr_warmup_ratio = 1.0 }
		$lr_warmup_steps = [int]([math]::Round($max_train_steps * $lr_warmup_ratio))
        $run_parameters += " --lr_warmup_steps=$lr_warmup_steps"
	}
    if ($min_snr_gamma -ne "0") { $run_parameters += " --min_snr_gamma=$min_snr_gamma" }
    if ($min_timestep -ne "0" ) { $run_parameters += " --min_timestep=$min_timestep" } 
	$run_parameters += " --network_dim=$network_dim"
	if ($network_alpha -ne 1) { $run_parameters += " --network_alpha=$network_alpha" }
    if ($network_dropout -gt 0) { $run_parameters += " --network_dropout=$network_dropout" }
    if ($scale_weight_norms -gt 0) { $run_parameters += " --scale_weight_norms=$scale_weight_norms" }
    if ($noise_offset -gt 0) { $run_parameters += " --noise_offset=$noise_offset"}
    if ($multires_noise_discount -gt 0) { $run_parameters += " --multires_noise_discount=$multires_noise_discount" }
    if ($multires_noise_discount -gt 0 -and $multires_noise_iterations -gt 0) { $run_parameters += " --multires_noise_iterations=$multires_noise_iterations" }
    if ($ip_noise_gamma -gt 0) { $run_parameters += " --ip_noise_gamma=$ip_noise_gamma" }
    if ($debiased_estimation_loss -eq 1 ) { $run_parameters += " --debiased_estimation_loss" }
    if ($zero_terminal_snr -eq 1 ) { $run_parameters += " --zero_terminal_snr" } 
    if ($train_with -eq 1) { $run_parameters += " --network_train_unet_only" }
    if ($train_with -eq 0) { $run_parameters += " --network_train_text_encoder_only" }
	if ($is_random_seed -le 0) { $seed = 1337 } 
	else { $seed = Get-Random }
	$run_parameters += " --seed=$seed"
	if ($shuffle_caption -ge 1) { $run_parameters += " --shuffle_caption" }
	$run_parameters += " --keep_tokens=$keep_tokens"
	
	# additional
	<# $run_parameters += " --device=`"$device`"" #> # wtf why ?
	if ($gradient_checkpointing -ge 1) { $run_parameters += " --gradient_checkpointing"  }
	if ($gradient_accumulation_steps -gt 1) { $run_parameters += " --gradient_accumulation_steps=$gradient_accumulation_steps" }
	$run_parameters += " --max_data_loader_n_workers=$max_data_loader_n_workers"
	if ($mixed_precision -eq "fp16" -or $mixed_precision -eq "bf16" -or $mixed_precision -eq "fp8") { $run_parameters += " --mixed_precision=$mixed_precision" }
	if ($save_precision -eq "float" -or $save_precision -eq "fp16" -or $save_precision -eq "bf16" -or $save_precision -eq "fp8") { $run_parameters += " --save_precision=$save_precision" }
	if ($logging_dir -ne "") { $run_parameters += " --logging_dir=`"$logging_dir`" --log_prefix=`"$log_prefix`"" }
	if ($debug_dataset -ge 1) { $run_parameters += " --debug_dataset" }
    if ($random_crop -eq 1) { $run_parameters += " --random_crop" }
    if ($cache_latents_to_disk -eq 1) { $run_parameters += " --cache_latents" }
    if ($cache_latents_to_disk -eq 1 -and $random_crop -eq 0 -and $cache_latents -eq 1) { $run_parameters += " --cache_latents_to_disk" }
    if ($bucket_no_upscale -eq 1) { $run_parameters += " --bucket_no_upscale --bucket_reso_steps=$bucket_reso_steps"}
    else { $run_parameters += " --min_bucket_reso=$min_bucket_reso --max_bucket_reso=$max_bucket_reso"}
    if ($no_metadata -eq 1) { $run_parameters += " --no_metadata" }
    if ($save_state -eq 1) { $run_parameters += " --save_state"}
	
	$run_parameters += " --caption_extension=`".txt`" --prior_loss_weight=1 --max_grad_norm=1.0 --enable_bucket --xformers --save_model_as=safetensors"
	
	if ($TestRun -ge 1) { $test_run = 1 }
	
	# main script
	if ($abort_script -ne "y") {
		sleep -s 0.3
		WCO black green 0 "Launching script with parameters:"
		sleep -s 0.3
		Write-Output "$($run_parameters -split '--' | foreach { if ($_ -ceq '') { Write-Output '' } else { Write-Output --`"$_`n`" } } | foreach { $_ -replace '=', ' = ' })"
		if ($test_run -le 0) {
			Set-Location -Path $sd_scripts_dir
			.\venv\Scripts\activate
			powershell accelerate launch --num_cpu_threads_per_process 2 train_network.py $run_parameters
            Start-Sleep -Seconds 1
		}
	}


# chain
if ($restart -ne 1 -and $abort_script -ne "y") { foreach ($script_string in $script_paths) {
	$path = $script_string -replace "^[ \t]+|[ \t]+$"
	if ($path -ne "" -and $path -match "^(?:[a-zA-Z]:[\\\/]|\.[\\\/])(?:[^\\\/:*?`"<>|+][^^:*?`"<>|+]+[^.][\\\/])+[^:\\*?`"<>|+]+(?:[^.:\\*?`"<>|+]+)$")
	{
		if (Test-Path -Path $path -PathType "leaf") {
			if ([System.IO.Path]::GetExtension($path) -eq ".ps1") {
				if ($TestRun -ge 1) {
					Write-Output "Launching next script in a chain (test run): $path"
					powershell -ChainedRun 1 -TestRun 1 -File $path }
				else {
					Write-Output "Launching next script in a chain: $path"
					powershell -ChainedRun 1 -File $path }
			}
			else { WCO black red 0 "Error: $path is not a valid script file" }
		}
		else { WCO black red 0 "Error: $path is not a file" }
	}
} }

# Autism case #2
Write-Output ""
if ($dont_draw_flags -le 0) {
$strl = 0
$version_string_length = $version_string.Length
while ($strl -lt ($([system.console]::BufferWidth))) { $strl += 1; WCO white white 1 " " }; Write-Output ""; $strl = 0; while ($version_string_length -lt $(($([system.console]::BufferWidth) + $version_string.Length) / 2)) { WCO darkblue white 1 " "; $version_string_length += 1 }; WCO darkblue white 1 $version_string; $version_string_length = $version_string.Length; while ($version_string_length -lt $(($([system.console]::BufferWidth) + $version_string.Length) / 2 - $version_string.Length % 2 + $([system.console]::BufferWidth) % 2)) { WCO darkblue white 1 " "; $version_string_length += 1 }; while ($strl -lt ($([system.console]::BufferWidth))) { $strl += 1; WCO darkred white 1 " " }
Write-Output "`n" }
sleep 3

if ($restart -eq 1) { powershell -File $PSCommandPath }

