// tests/einsum_compile_errors.rs — compile-fail tests for einsum! diagnostics.

#[test]
fn einsum_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/einsum_errors/*.rs");
}
