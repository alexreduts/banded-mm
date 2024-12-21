"""
Copyright [2024] [Alex Studer]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def BdGEMM(
        C: np.ndarray,
        A: np.ndarray,
        kl_A: int,
        ku_A: int,
        B: np.ndarray,
        kl_B: int,
        ku_B: int,
        block_size_outer,
        block_size_inner,
        **kwargs
        ):
    match(implementation):
        case 'streamed':
            print("Implementation specifiec: 'streamed'")
            BdGEMM_streamed(C, A, kl_A, ku_A, B, kl_B, ku_B, block_size_outer, block_size_inner)
        case 'blocking':
            print("Implementation specifice: 'blocking'")
            BdGEMM_blocking(C, A, kl_A, ku_A, B, kl_B, ku_B, block_size_outer, block_size_inner)
        else 'naiveCopy':
            BdGEMM_naiveCopy(C, kl_A, ku_A, B, kl_B, ku_B, block_size_outer, block_size_inner)
    
def BdMM(
        A: np.ndarray,
        ku_A: int,
        kl_A: int,
        B: np.ndarray,
        ku_B: int,
        kl_B: int
        ):
    BdMM_naiveCopy(A, ku_A, kl_A, B, ku_B, kl_B)