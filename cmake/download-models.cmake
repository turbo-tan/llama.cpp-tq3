get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

set(_download_ok FALSE)
set(_download_attempts 5)
set(_download_url "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true")

foreach(_attempt RANGE 1 ${_download_attempts})
    file(DOWNLOAD
        "${_download_url}"
        "${DEST}"
        TLS_VERIFY ON
        EXPECTED_HASH ${HASH}
        STATUS status
    )

    list(GET status 0 code)
    if(code EQUAL 0)
        set(_download_ok TRUE)
        break()
    endif()

    list(GET status 1 msg)
    if(EXISTS "${DEST}")
        file(REMOVE "${DEST}")
    endif()

    if(_attempt LESS _download_attempts)
        message(WARNING "Download attempt ${_attempt}/${_download_attempts} for ${NAME} failed: ${msg}; retrying...")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 2)
    endif()
endforeach()

if(NOT _download_ok)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
