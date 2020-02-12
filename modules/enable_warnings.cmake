if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE
            -Werror
            -Wall
            -Wextra
            -Waddress
            -Warray-bounds
            -Wbuiltin-macro-redefined
            -Wswitch
            -Wunreachable-code
            -pedantic
            -pedantic-errors)
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE
            -Werror
            -Weverything
            -Wno-padded
            -Wno-shadow
            -Wno-language-extension-token
            -Wno-reserved-id-macro
            -Wno-missing-noreturn
            -Wno-deprecated-declarations)

endif ()
