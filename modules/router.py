from fastapi import APIRouter, status
from .scheme import KernelPCAIn, KernelPCAOut
from kpca import KernelPCA


modules_router = APIRouter()


kernel_pca = KernelPCA(
    gamma=.1,
    n_components=1
)


@modules_router.post(
    '/',
    summary='Kernel principal component analysis fit',
    status_code=status.HTTP_200_OK,
    response_model=KernelPCAOut
)
async def kpca_fit(
        kernel_pca_scheme: KernelPCAIn
) -> KernelPCAOut:
    kernel_pca_scheme = KernelPCAOut(
        X = kernel_pca.fit(kernel_pca_scheme.X).tolist()
    )
    return kernel_pca_scheme