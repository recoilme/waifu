# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma

#ASPECT_RATIO_384 = {}
#width = 192
#height = 768
#step = 64

# Перебираем все возможные значения ширины и высоты с шагом 64
#for w in range(width, height + 1, step):  # Диапазон ширины
#    for h in range(width, height + 1, step):  # Диапазон высоты
#        ratio = round(w / h, 2)  # Вычисляем соотношение сторон и округляем до 2 знаков
#        ASPECT_RATIO_384[str(ratio)] = [float(w), float(h)]  # Добавляем в словарь

# Отсортировать словарь по ключу
#ASPECT_RATIO_384 = dict(sorted(ASPECT_RATIO_384.items()))

# Вывод словаря в нужном формате
#output = "ASPECT_RATIO_384 = {\n"
#for key, value in ASPECT_RATIO_384.items():
#    output += f'    "{key}": [{value[0]}, {value[1]}],\n'
#output += "}"

#print(output)

ASPECT_RATIO_4096 = {
    "0.25": [2048.0, 8192.0],
    "0.26": [2048.0, 7936.0],
    "0.27": [2048.0, 7680.0],
    "0.28": [2048.0, 7424.0],
    "0.32": [2304.0, 7168.0],
    "0.33": [2304.0, 6912.0],
    "0.35": [2304.0, 6656.0],
    "0.4": [2560.0, 6400.0],
    "0.42": [2560.0, 6144.0],
    "0.48": [2816.0, 5888.0],
    "0.5": [2816.0, 5632.0],
    "0.52": [2816.0, 5376.0],
    "0.57": [3072.0, 5376.0],
    "0.6": [3072.0, 5120.0],
    "0.68": [3328.0, 4864.0],
    "0.72": [3328.0, 4608.0],
    "0.78": [3584.0, 4608.0],
    "0.82": [3584.0, 4352.0],
    "0.88": [3840.0, 4352.0],
    "0.94": [3840.0, 4096.0],
    "1.0": [4096.0, 4096.0],
    "1.07": [4096.0, 3840.0],
    "1.13": [4352.0, 3840.0],
    "1.21": [4352.0, 3584.0],
    "1.29": [4608.0, 3584.0],
    "1.38": [4608.0, 3328.0],
    "1.46": [4864.0, 3328.0],
    "1.67": [5120.0, 3072.0],
    "1.75": [5376.0, 3072.0],
    "2.0": [5632.0, 2816.0],
    "2.09": [5888.0, 2816.0],
    "2.4": [6144.0, 2560.0],
    "2.5": [6400.0, 2560.0],
    "2.89": [6656.0, 2304.0],
    "3.0": [6912.0, 2304.0],
    "3.11": [7168.0, 2304.0],
    "3.62": [7424.0, 2048.0],
    "3.75": [7680.0, 2048.0],
    "3.88": [7936.0, 2048.0],
    "4.0": [8192.0, 2048.0],
}

ASPECT_RATIO_2880 = {
    "0.25": [1408.0, 5760.0],
    "0.26": [1408.0, 5568.0],
    "0.27": [1408.0, 5376.0],
    "0.28": [1408.0, 5184.0],
    "0.32": [1600.0, 4992.0],
    "0.33": [1600.0, 4800.0],
    "0.34": [1600.0, 4672.0],
    "0.4": [1792.0, 4480.0],
    "0.42": [1792.0, 4288.0],
    "0.47": [1920.0, 4096.0],
    "0.49": [1920.0, 3904.0],
    "0.51": [1920.0, 3776.0],
    "0.55": [2112.0, 3840.0],
    "0.59": [2112.0, 3584.0],
    "0.68": [2304.0, 3392.0],
    "0.72": [2304.0, 3200.0],
    "0.78": [2496.0, 3200.0],
    "0.83": [2496.0, 3008.0],
    "0.89": [2688.0, 3008.0],
    "0.93": [2688.0, 2880.0],
    "1.0": [2880.0, 2880.0],
    "1.07": [2880.0, 2688.0],
    "1.12": [3008.0, 2688.0],
    "1.21": [3008.0, 2496.0],
    "1.28": [3200.0, 2496.0],
    "1.39": [3200.0, 2304.0],
    "1.47": [3392.0, 2304.0],
    "1.7": [3584.0, 2112.0],
    "1.82": [3840.0, 2112.0],
    "2.03": [3904.0, 1920.0],
    "2.13": [4096.0, 1920.0],
    "2.39": [4288.0, 1792.0],
    "2.5": [4480.0, 1792.0],
    "2.92": [4672.0, 1600.0],
    "3.0": [4800.0, 1600.0],
    "3.12": [4992.0, 1600.0],
    "3.68": [5184.0, 1408.0],
    "3.82": [5376.0, 1408.0],
    "3.95": [5568.0, 1408.0],
    "4.0": [5760.0, 1408.0],
}

ASPECT_RATIO_2048 = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.27": [1024.0, 3840.0],
    "0.28": [1024.0, 3712.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "2.89": [3328.0, 1152.0],
    "3.0": [3456.0, 1152.0],
    "3.11": [3584.0, 1152.0],
    "3.62": [3712.0, 1024.0],
    "3.75": [3840.0, 1024.0],
    "3.88": [3968.0, 1024.0],
    "4.0": [4096.0, 1024.0],
}

ASPECT_RATIO_1024 = {
    "0.25": [512.0, 2048.0],
    "0.26": [512.0, 1984.0],
    "0.27": [512.0, 1920.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "2.89": [1664.0, 576.0],
    "3.0": [1728.0, 576.0],
    "3.11": [1792.0, 576.0],
    "3.62": [1856.0, 512.0],
    "3.75": [1920.0, 512.0],
    "3.88": [1984.0, 512.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_510 = {
    "0.25": [256.0, 1024.0],
    "0.26": [256.0, 992.0],
    "0.27": [256.0, 960.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "2.89": [832.0, 288.0],
    "3.0": [864.0, 288.0],
    "3.11": [896.0, 288.0],
    "3.62": [928.0, 256.0],
    "3.75": [960.0, 256.0],
    "3.88": [992.0, 256.0],
    "4.0": [1024.0, 256.0],
}

ASPECT_RATIO_256 = {
    "0.25": [128.0, 512.0],
    "0.26": [128.0, 496.0],
    "0.27": [128.0, 480.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "2.89": [416.0, 144.0],
    "3.0": [432.0, 144.0],
    "3.11": [448.0, 144.0],
    "3.62": [464.0, 128.0],
    "3.75": [480.0, 128.0],
    "3.88": [496.0, 128.0],
    "4.0": [512.0, 128.0],
}

ASPECT_RATIO_256_TEST = {
    "0.25": [128.0, 512.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "3.0": [432.0, 144.0],
    "4.0": [512.0, 128.0],
}

ASPECT_RATIO_384 = {
    "0.25": [192.0, 768.0],
    "0.27": [192.0, 704.0],
    "0.3": [192.0, 640.0],
    "0.33": [256.0, 768.0],
    "0.36": [256.0, 704.0],
    "0.38": [192.0, 512.0],
    "0.4": [256.0, 640.0],
    "0.42": [320.0, 768.0],
    "0.43": [192.0, 448.0],
    "0.44": [256.0, 576.0],
    "0.45": [320.0, 704.0],
    "0.5": [384.0, 768.0],
    "0.55": [384.0, 704.0],
    "0.56": [320.0, 576.0],
    "0.57": [256.0, 448.0],
    "0.58": [448.0, 768.0],
    "0.6": [384.0, 640.0],
    "0.62": [320.0, 512.0],
    "0.64": [448.0, 704.0],
    "0.67": [512.0, 768.0],
    "0.7": [448.0, 640.0],
    "0.71": [320.0, 448.0],
    "0.73": [512.0, 704.0],
    "0.75": [576.0, 768.0],
    "0.78": [448.0, 576.0],
    "0.8": [512.0, 640.0],
    "0.82": [576.0, 704.0],
    "0.83": [640.0, 768.0],
    "0.86": [384.0, 448.0],
    "0.88": [448.0, 512.0],
    "0.89": [512.0, 576.0],
    "0.9": [576.0, 640.0],
    "0.91": [640.0, 704.0],
    "0.92": [704.0, 768.0],
    "1.0": [768.0, 768.0],
    "1.09": [768.0, 704.0],
    "1.1": [704.0, 640.0],
    "1.11": [640.0, 576.0],
    "1.12": [576.0, 512.0],
    "1.14": [512.0, 448.0],
    "1.17": [448.0, 384.0],
    "1.2": [768.0, 640.0],
    "1.22": [704.0, 576.0],
    "1.25": [640.0, 512.0],
    "1.29": [576.0, 448.0],
    "1.33": [768.0, 576.0],
    "1.38": [704.0, 512.0],
    "1.4": [448.0, 320.0],
    "1.43": [640.0, 448.0],
    "1.5": [768.0, 512.0],
    "1.57": [704.0, 448.0],
    "1.6": [512.0, 320.0],
    "1.67": [640.0, 384.0],
    "1.71": [768.0, 448.0],
    "1.75": [448.0, 256.0],
    "1.8": [576.0, 320.0],
    "1.83": [704.0, 384.0],
    "2.0": [768.0, 384.0],
    "2.2": [704.0, 320.0],
    "2.25": [576.0, 256.0],
    "2.33": [448.0, 192.0],
    "2.4": [768.0, 320.0],
    "2.5": [640.0, 256.0],
    "2.67": [512.0, 192.0],
    "2.75": [704.0, 256.0],
    "3.0": [768.0, 256.0],
    "3.33": [640.0, 192.0],
    "3.67": [704.0, 192.0],
    "4.0": [768.0, 192.0],
}

ASPECT_RATIO_384_TEST = {
    "0.25": [192.0, 768.0],
    "0.27": [192.0, 704.0],
    "0.3": [192.0, 640.0],
    "0.33": [256.0, 768.0],
    "0.36": [256.0, 704.0],
    "0.38": [192.0, 512.0],
    "0.4": [256.0, 640.0],
    "0.42": [320.0, 768.0],
    "0.43": [192.0, 448.0],
    "0.44": [256.0, 576.0],
    "0.45": [320.0, 704.0],
    "0.5": [384.0, 768.0],
    "0.55": [384.0, 704.0],
    "0.56": [320.0, 576.0],
    "0.57": [256.0, 448.0],
    "0.58": [448.0, 768.0],
    "0.6": [384.0, 640.0],
    "0.62": [320.0, 512.0],
    "0.64": [448.0, 704.0],
    "0.67": [512.0, 768.0],
    "0.7": [448.0, 640.0],
    "0.71": [320.0, 448.0],
    "0.73": [512.0, 704.0],
    "0.75": [576.0, 768.0],
    "0.78": [448.0, 576.0],
    "0.8": [512.0, 640.0],
    "0.82": [576.0, 704.0],
    "0.83": [640.0, 768.0],
    "0.86": [384.0, 448.0],
    "0.88": [448.0, 512.0],
    "0.89": [512.0, 576.0],
    "0.9": [576.0, 640.0],
    "0.91": [640.0, 704.0],
    "0.92": [704.0, 768.0],
    "1.0": [768.0, 768.0],
    "1.09": [768.0, 704.0],
    "1.1": [704.0, 640.0],
    "1.11": [640.0, 576.0],
    "1.12": [576.0, 512.0],
    "1.14": [512.0, 448.0],
    "1.17": [448.0, 384.0],
    "1.2": [768.0, 640.0],
    "1.22": [704.0, 576.0],
    "1.25": [640.0, 512.0],
    "1.29": [576.0, 448.0],
    "1.33": [768.0, 576.0],
    "1.38": [704.0, 512.0],
    "1.4": [448.0, 320.0],
    "1.43": [640.0, 448.0],
    "1.5": [768.0, 512.0],
    "1.57": [704.0, 448.0],
    "1.6": [512.0, 320.0],
    "1.67": [640.0, 384.0],
    "1.71": [768.0, 448.0],
    "1.75": [448.0, 256.0],
    "1.8": [576.0, 320.0],
    "1.83": [704.0, 384.0],
    "2.0": [768.0, 384.0],
    "2.2": [704.0, 320.0],
    "2.25": [576.0, 256.0],
    "2.33": [448.0, 192.0],
    "2.4": [768.0, 320.0],
    "2.5": [640.0, 256.0],
    "2.67": [512.0, 192.0],
    "2.75": [704.0, 256.0],
    "3.0": [768.0, 256.0],
    "3.33": [640.0, 192.0],
    "3.67": [704.0, 192.0],
    "4.0": [768.0, 192.0],
}

ASPECT_RATIO_510_TEST = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}

ASPECT_RATIO_512 = {
    "0.5": [384.0, 768.0],
    "0.55": [384.0, 704.0],
    "0.58": [448.0, 768.0],
    "0.6": [384.0, 640.0],
    "0.64": [448.0, 704.0],
    "0.67": [512.0, 768.0],
    "0.7": [448.0, 640.0],
    "0.73": [512.0, 704.0],
    "0.75": [576.0, 768.0],
    "0.78": [448.0, 576.0],
    "0.8": [512.0, 640.0],
    "0.82": [576.0, 704.0],
    "0.83": [640.0, 768.0],
    "0.86": [384.0, 448.0],
    "0.88": [448.0, 512.0],
    "0.89": [512.0, 576.0],
    "0.9": [576.0, 640.0],
    "0.91": [640.0, 704.0],
    "0.92": [704.0, 768.0],
    "1.0": [768.0, 768.0],
    "1.09": [768.0, 704.0],
    "1.1": [704.0, 640.0],
    "1.11": [640.0, 576.0],
    "1.12": [576.0, 512.0],
    "1.14": [512.0, 448.0],
    "1.17": [448.0, 384.0],
    "1.2": [768.0, 640.0],
    "1.22": [704.0, 576.0],
    "1.25": [640.0, 512.0],
    "1.29": [576.0, 448.0],
    "1.33": [768.0, 576.0],
    "1.38": [704.0, 512.0],
    "1.43": [640.0, 448.0],
    "1.5": [768.0, 512.0],
    "1.57": [704.0, 448.0],
    "1.67": [640.0, 384.0],
    "1.71": [768.0, 448.0],
    "1.83": [704.0, 384.0],
    "2.0": [768.0, 384.0],
}

ASPECT_RATIO_512_TEST = {
    "0.5": [384.0, 768.0],
    "0.55": [384.0, 704.0],
    "0.58": [448.0, 768.0],
    "0.6": [384.0, 640.0],
    "0.64": [448.0, 704.0],
    "0.67": [512.0, 768.0],
    "0.7": [448.0, 640.0],
    "0.73": [512.0, 704.0],
    "0.75": [576.0, 768.0],
    "0.78": [448.0, 576.0],
    "0.8": [512.0, 640.0],
    "0.82": [576.0, 704.0],
    "0.83": [640.0, 768.0],
    "0.86": [384.0, 448.0],
    "0.88": [448.0, 512.0],
    "0.89": [512.0, 576.0],
    "0.9": [576.0, 640.0],
    "0.91": [640.0, 704.0],
    "0.92": [704.0, 768.0],
    "1.0": [768.0, 768.0],
    "1.09": [768.0, 704.0],
    "1.1": [704.0, 640.0],
    "1.11": [640.0, 576.0],
    "1.12": [576.0, 512.0],
    "1.14": [512.0, 448.0],
    "1.17": [448.0, 384.0],
    "1.2": [768.0, 640.0],
    "1.22": [704.0, 576.0],
    "1.25": [640.0, 512.0],
    "1.29": [576.0, 448.0],
    "1.33": [768.0, 576.0],
    "1.38": [704.0, 512.0],
    "1.43": [640.0, 448.0],
    "1.5": [768.0, 512.0],
    "1.57": [704.0, 448.0],
    "1.67": [640.0, 384.0],
    "1.71": [768.0, 448.0],
    "1.83": [704.0, 384.0],
    "2.0": [768.0, 384.0],
}


ASPECT_RATIO_1024_TEST = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_2048_TEST = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "3.0": [3456.0, 1152.0],
    "4.0": [4096.0, 1024.0],
}

ASPECT_RATIO_2880_TEST = {
    "0.25": [2048.0, 8192.0],
    "0.26": [2048.0, 7936.0],
    "0.32": [2304.0, 7168.0],
    "0.33": [2304.0, 6912.0],
    "0.35": [2304.0, 6656.0],
    "0.4": [2560.0, 6400.0],
    "0.42": [2560.0, 6144.0],
    "0.48": [2816.0, 5888.0],
    "0.5": [2816.0, 5632.0],
    "0.52": [2816.0, 5376.0],
    "0.57": [3072.0, 5376.0],
    "0.6": [3072.0, 5120.0],
    "0.68": [3328.0, 4864.0],
    "0.72": [3328.0, 4608.0],
    "0.78": [3584.0, 4608.0],
    "0.82": [3584.0, 4352.0],
    "0.88": [3840.0, 4352.0],
    "0.94": [3840.0, 4096.0],
    "1.0": [4096.0, 4096.0],
    "1.07": [4096.0, 3840.0],
    "1.13": [4352.0, 3840.0],
    "1.21": [4352.0, 3584.0],
    "1.29": [4608.0, 3584.0],
    "1.38": [4608.0, 3328.0],
    "1.46": [4864.0, 3328.0],
    "1.67": [5120.0, 3072.0],
    "1.75": [5376.0, 3072.0],
    "2.0": [5632.0, 2816.0],
    "2.09": [5888.0, 2816.0],
    "2.4": [6144.0, 2560.0],
    "2.5": [6400.0, 2560.0],
    "3.0": [6912.0, 2304.0],
    "4.0": [8192.0, 2048.0],
}

ASPECT_RATIO_4096_TEST = {
    "0.25": [2048.0, 8192.0],
    "0.26": [2048.0, 7936.0],
    "0.27": [2048.0, 7680.0],
    "0.28": [2048.0, 7424.0],
    "0.32": [2304.0, 7168.0],
    "0.33": [2304.0, 6912.0],
    "0.35": [2304.0, 6656.0],
    "0.4": [2560.0, 6400.0],
    "0.42": [2560.0, 6144.0],
    "0.48": [2816.0, 5888.0],
    "0.5": [2816.0, 5632.0],
    "0.52": [2816.0, 5376.0],
    "0.57": [3072.0, 5376.0],
    "0.6": [3072.0, 5120.0],
    "0.68": [3328.0, 4864.0],
    "0.72": [3328.0, 4608.0],
    "0.78": [3584.0, 4608.0],
    "0.82": [3584.0, 4352.0],
    "0.88": [3840.0, 4352.0],
    "0.94": [3840.0, 4096.0],
    "1.0": [4096.0, 4096.0],
    "1.07": [4096.0, 3840.0],
    "1.13": [4352.0, 3840.0],
    "1.21": [4352.0, 3584.0],
    "1.29": [4608.0, 3584.0],
    "1.38": [4608.0, 3328.0],
    "1.46": [4864.0, 3328.0],
    "1.67": [5120.0, 3072.0],
    "1.75": [5376.0, 3072.0],
    "2.0": [5632.0, 2816.0],
    "2.09": [5888.0, 2816.0],
    "2.4": [6144.0, 2560.0],
    "2.5": [6400.0, 2560.0],
    "2.89": [6656.0, 2304.0],
    "3.0": [6912.0, 2304.0],
    "3.11": [7168.0, 2304.0],
    "3.62": [7424.0, 2048.0],
    "3.75": [7680.0, 2048.0],
    "3.88": [7936.0, 2048.0],
    "4.0": [8192.0, 2048.0],
}

ASPECT_RATIO_1280_TEST = {"1.0": [1280.0, 1280.0]}
ASPECT_RATIO_1536_TEST = {"1.0": [1536.0, 1536.0]}
ASPECT_RATIO_768_TEST = {"1.0": [768.0, 768.0]}


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
