#![cfg(feature = "cpu")]

#[test]
fn einsum_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/einsum_errors/*.rs");
}
