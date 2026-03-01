#![cfg(feature = "cpu")]

#[test]
fn einsum_compile_errors() {
    trybuild::TestCases::new().compile_fail("tests/einsum_errors/*.rs");
}
