# coding: utf-8

"""Re-export of the single Cyberwave exception hierarchy.

Written by ``python-sdk-gen.sh`` — do not edit. The OpenAPI generator overwrites
this whole directory, and the generator's own copy of the ``ApiException`` family
is replaced by this file so that ``cyberwave.rest.exceptions.UnauthorizedException``
and ``cyberwave.exceptions.UnauthorizedException`` are the same class.

Import from ``cyberwave.exceptions`` in new code; this module exists so the
generated client keeps working against the path it was generated to import.
"""

from cyberwave.exceptions import (
    ApiAttributeError as ApiAttributeError,
    ApiException as ApiException,
    ApiKeyError as ApiKeyError,
    ApiTypeError as ApiTypeError,
    ApiValueError as ApiValueError,
    BadRequestException as BadRequestException,
    ConflictException as ConflictException,
    ForbiddenException as ForbiddenException,
    NotFoundException as NotFoundException,
    OpenApiException as OpenApiException,
    PaymentRequiredException as PaymentRequiredException,
    ServiceException as ServiceException,
    UnauthorizedException as UnauthorizedException,
    UnprocessableEntityException as UnprocessableEntityException,
    render_path as render_path,
)

__all__ = [
    "ApiAttributeError",
    "ApiException",
    "ApiKeyError",
    "ApiTypeError",
    "ApiValueError",
    "BadRequestException",
    "ConflictException",
    "ForbiddenException",
    "NotFoundException",
    "OpenApiException",
    "PaymentRequiredException",
    "ServiceException",
    "UnauthorizedException",
    "UnprocessableEntityException",
    "render_path",
]
