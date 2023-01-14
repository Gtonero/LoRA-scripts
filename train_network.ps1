# https://github.com/cloneofsimo/lora
# https://github.com/kohya-ss/sd-scripts
# https://rentry.org/2chAI_LoRA_Dreambooth_guide

##### Начало конфига #####

$sd_scripts_dir = "X:\git-repos\sd-scripts\" # Путь к папке с репозиторием kohya-ss/sd-scripts

$ckpt = "X:\SD-models\checkpoint.safetensors" # Путь к чекпоинту (ckpt / safetensors)
$is_sd_v2_ckpt = 0 # Поставь '1' если загружаешь SD 2.x чекпоинт
$is_sd_v2_768_ckpt = 0 # Также поставь здесь значение '1', если загружаешь SD 2.x-768 чекпоинт
$image_dir = "X:\training_data\img\" # Путь к папке с изображениями
$reg_dir = "X:\training_data\img_reg\" # Путь к папке с регуляризационными изображениями (можно указать на пустую папку, но путь обязательно должен быть указан)
$output_dir = "X:\LoRA\" # Директория сохранения LoRA чекпоинтов
$output_name = "my_LoRA_network_v1" # Название файла (расширение не нужно)

$train_batch_size = 1 # Сколько изображений тренировать одновременно. Чем больше значение, тем быстрее тренировка, но больше потребление видеопамяти
$resolution = 512 # Разрешение тренировки
$num_epochs = 10 # Число эпох
$save_every_n_epochs = 1 # Сохранять чекпоинт каждые n эпох
$save_last_n_epochs = 999 # Сохранить только последние n эпох
$max_token_length = 75 # Максимальная длина токена. Возможные значения: 75 / 150 / 225
$clip_skip = 1 # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#ignore-last-layers-of-clip-model

# Дополнительные настройки, можно оставить по умолчанию

$learning_rate = 1e-4 # Скорость обучения
$unet_lr = $learning_rate # Скорость обучения U-Net
$text_encoder_lr = $learning_rate # Скорость обучения текстового энкодера
$scheduler = "cosine_with_restarts" # linear, cosine, cosine_with_restarts, polynomial, constant (по умолчанию), constant_with_warmup
$network_dim = 128 # Размер нетворка. Чем больше значение, тем больше точность и размер выходного файла
$save_precision = "fp16" # None, float, fp16, bf16
$mixed_precision = "fp16" # no, fp16, bf16
$is_random_seed = 1 # 1 -- рандомный сид, 0 -- статичный
$shuffle_caption = 1 # Перемешивать файлы описания, затеганные через запятую
$use_vae = 0 # Использовать VAE
$vae_path = "X:\SD-models\checkpoint.vae.pt" # Путь к VAE

# Логгирование

$logging_enabled = 0
$logging_dir = "X:\LoRA\logs\"
$log_prefix = $output_name

##### Конец конфига #####

function Is-Numeric ($value) { return $value -match "^[\d\.]+$" }

function Write-ColorOutput($ForegroundColor)
{
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) { Write-Output $args }
    else { $input | Write-Output }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Word-Ending($value)
{
	$ending = $value.ToString()
	if ($ending -ge "11" -and $ending -le "19") { return "й" }
	$ending = $ending.Substring([Math]::Max($ending.Length, 0) - 1)
	if ($ending -eq "1") { return "е" }
	if ($ending -ge "2" -and $ending -le "4") { return "я" }
	if (($ending -ge "5" -and $ending -le "9") -or $ending -eq "0") { return "й" }
}

Write-Output "Подсчет количества изображений в папках"
$total = 0
$is_structure_wrong = 0
$abort_script = 0
$iter = 0

Get-ChildItem -Path $image_dir -Directory | ForEach-Object {
    $parts = $_.Name.Split("_")
    if (!(Is-Numeric $parts[0]))
    {
		Write-ColorOutput red "Ошибка в $($_):`n`t$($parts[0]) не является числом"
		$is_structure_wrong = 1
        return
    }
	if ([int]$parts[0] -le 0)
	{
		Write-ColorOutput red "Ошибка в $($_):`nПовторения в имени папки с изображениями должно быть >0"
		$is_structure_wrong = 1
        return
	}
    $repeats = [int]$parts[0]
    $imgs = Get-ChildItem $_.FullName -Depth 0 -File -Include *.jpg, *.png, *.webp | Measure-Object | ForEach-Object { $_.Count }
	if ($iter -eq 0) { Write-Output "Обучающие изображения:" }
    $img_repeats = ($repeats * $imgs)
    Write-Output "`t$($parts[1]): $repeats повторени$(Word-Ending $repeats) * $imgs изображени$(Word-Ending $imgs) = $($img_repeats)"
    $total += $img_repeats
	$iter += 1
}

$iter = 0

if ($is_structure_wrong -eq 0) { Get-ChildItem -Path $reg_dir -Directory | % { if ($abort_script -ne "n") { ForEach-Object {
    $parts = $_.Name.Split("_")
    if (!(Is-Numeric $parts[0]))
    {
		Write-ColorOutput red "Ошибка в $($_):`n`t$($parts[0]) не является числом"
		$is_structure_wrong = 1
        return
    }
	if ([int]$parts[0] -le 0)
	{
		Write-ColorOutput red "Ошибка в $($_):`nПовторения в имени папки с изображениями должно быть >0"
		$is_structure_wrong = 1
        return
	}
    $repeats = [int]$parts[0]
    $reg_imgs = Get-ChildItem $_.FullName -Depth 0 -File -Include *.jpg, *.png, *.webp | Measure-Object | ForEach-Object { $_.Count }
	if ($iter -eq 0) { Write-Output "Регуляризационные изображения:" }
	if ($reg_imgs -eq 0)
	{
		Write-ColorOutput darkyellow "Внимание: папка для регуляризационных изображений присутствует, но в ней ничего нет"
		do { $abort_script = Read-Host "Прервать выполнение скрипта? (y/N)" }
		until (($abort_script -eq "y") -or ($abort_script -ceq "N"))
		return
	}
	else
	{
		$img_repeats = ($repeats * $reg_imgs)
		Write-Output "`t$($parts[1]): $repeats повторени$(Word-Ending $repeats) * $reg_imgs изображени$(Word-Ending $reg_imgs) = $($img_repeats)"
		$iter += 1
	}
} } } }

if ($is_structure_wrong -eq 0 -and ($abort_script -eq "n" -or $abort_script -eq 0))
{
	if ($reg_imgs -gt 0)
	{
		$total *= 2
		Write-Output "Количество шагов увеличено вдвое: количество регуляризационных изображений больше 0"
	}
	
	Write-Output "Количество изображений с повторениями: $total"
	Write-Output "Размер обучающей партии (train_batch_size): $train_batch_size"
	Write-Output "Количество эпох: $num_epochs"
	$max_training_steps = [int]($total / $train_batch_size * $num_epochs)
	Write-Output "Количество шагов: $total / $train_batch_size * $num_epochs = $max_training_steps"
	
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
		else { Write-ColorOutput darkyellow "Неверно указан max_token_length! Используем значение 75" }
	}
	
	if ($is_sd_v2_ckpt -le 0) { Write-Output "Stable Diffusion 1.x чекпоинт" }
	if ($is_sd_v2_ckpt -ge 1)
	{
		if ($is_sd_v2_768_ckpt -ge 1)
		{
			$v2_resolution = "768"
			$run_parameters += " --v_parameterization"
		}
		else { $v2_resolution = "512" }
		Write-Output "Stable Diffusion 2.x ($v2_resolution) чекпоинт"
		$run_parameters += " --v2"
		if ($clip_skip -eq -not 1)
		{
			Write-ColorOutput darkyellow "Внимание: результаты тренировки SD 2.x чекпоинта с clip_skip отличным от 1 могут быть непредсказуемые"
			do { $abort_script = Read-Host "Прервать выполнение скрипта? (y/N)" }
			until (($abort_script -eq "y") -or ($abort_script -ceq "N"))
		}
	}
	
	if ($shuffle_caption -ge 1) { $run_parameters += " --shuffle_caption" }
	
	if ($logging_enabled -ge 1) { $run_parameters += " --logging_dir=`"$logging_dir`" --log_prefix=`"$output_name`""}
	
	if ($use_vae -ge 1) { $run_parameters += " --vae=`"$vae_path`"" }
	
	sleep -s 1
	
	if ($abort_script -eq "n" -or $abort_script -eq 0)
	{
		Write-ColorOutput green "Выполнение скрипта с параметрами:"
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