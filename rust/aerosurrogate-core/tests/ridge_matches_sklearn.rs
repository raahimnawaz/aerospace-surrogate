//! The baked-in Ridge surrogate must reproduce the trained sklearn pipeline
//! to ~machine precision. Five (features, sklearn_cl, sklearn_cd, sklearn_cm)
//! samples are baked into `surrogate::ridge` itself by the export script;
//! this test runs the Rust `predict` on each and asserts agreement to 1e-12.

use aerosurrogate_core::surrogate::ridge::{
    predict, N_SAMPLES, SAMPLE_CD, SAMPLE_CL, SAMPLE_CM, SAMPLE_X,
};

#[test]
fn ridge_predict_matches_sklearn_to_machine_precision() {
    for k in 0..N_SAMPLES {
        let p = predict(&SAMPLE_X[k]);
        let dcl = (p.cl - SAMPLE_CL[k]).abs();
        let dcd = (p.cd - SAMPLE_CD[k]).abs();
        let dcm = (p.cm - SAMPLE_CM[k]).abs();
        assert!(
            dcl < 1e-12,
            "sample {k}: CL diff {dcl:.3e} (rust {} vs sklearn {})",
            p.cl,
            SAMPLE_CL[k]
        );
        assert!(
            dcd < 1e-12,
            "sample {k}: CD diff {dcd:.3e} (rust {} vs sklearn {})",
            p.cd,
            SAMPLE_CD[k]
        );
        assert!(
            dcm < 1e-12,
            "sample {k}: CM diff {dcm:.3e} (rust {} vs sklearn {})",
            p.cm,
            SAMPLE_CM[k]
        );
    }
}
