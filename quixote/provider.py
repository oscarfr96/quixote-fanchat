"""Errores comprensibles, sin incluir mensajes que pudieran contener credenciales."""

from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError


class ProviderError(RuntimeError):
    pass


def call_api(operation, **kwargs):
    try:
        return operation(**kwargs)
    except AuthenticationError:
        raise ProviderError("OpenAI no ha aceptado la clave. Revisa OPENAI_API_KEY en tu .env.") from None
    except RateLimitError:
        raise ProviderError("OpenAI ha indicado un límite de peticiones o de saldo. Revisa tu cuenta e inténtalo después.") from None
    except APIConnectionError:
        raise ProviderError("No se ha podido conectar con OpenAI. Revisa la conexión a Internet.") from None
    except APIError as exc:
        raise ProviderError(
            f"OpenAI no pudo completar la petición (HTTP {getattr(exc, 'status_code', 'desconocido')}). "
            "Comprueba que el modelo de tu .env está disponible y admite Structured Outputs."
        ) from None
