//! A schema-3 bank file frozen at 0.2.0 (`docs/PLAN.md` tasks 33-35): the
//! first layout that carries the last learned row (task 34) and the data
//! summary (task 35) alongside the models, with `kalman` and `sgd` keeping
//! an `EwDiag` standardizer (task 33).
//!
//! `state_v1.rs` and `state_schema2.rs` prove that files written before
//! those tasks still load and continue; this file is the other half of the
//! promise, for the *next* layout change: what 0.2.0 wrote must keep loading
//! with its summary, its last row and its models intact, and must continue
//! the stream with the same bits as a bank that never left the build. The
//! fixture is frozen bytes for the same reasons as the others -- there will
//! be no schema-3 writer once the layout moves, and hard rule 1 keeps binary
//! files out of the repo -- and was generated once, from this build, by the
//! stream `frame` reproduces and the specs the file itself carries.
//!
//! When `SCHEMA_VERSION` moves, keep this file and its fixture as they are:
//! the tests already ask only for what a converter must preserve, and
//! `a_schema_3_state_re_saves_byte_identically` is written to hold while the
//! writer is unchanged and to become the upgrade test when it is not.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// A schema-3 bank state: specs `k` (`kalman`, standardized, weighted,
/// grouped) and `r` (`ew_ridge`, standardized, weighted, grouped, sessioned
/// with `session_gap = "reset"`), after `frame(0, 60)` in three chunks of
/// twenty. See the module docs for why this is frozen bytes.
const SCHEMA3_STATE_HEX: &str = concat!(
    "87a56d61676963b2706f6c6172732d6f6e6c696e652d62616e6bae666f726d61745f76657273",
    "696f6e02ae736368656d615f76657273696f6e03af7061636b6167655f76657273696f6ea530",
    "2e322e30a5737065637392de001fa46e616d65a16ba56d6f64656c88a474797065a66b616c6d",
    "616ead636f65665f68616c666c696665cb4049000000000000a171c0a76f62735f766172c0a2",
    "7030c0a773686172655f70c2af7265766572745f68616c666c696665c0ab7374616e64617264",
    "697a65c3a77461726765747391a179a8666561747572657392a27830a27831ad6164645f696e",
    "74657263657074c3a5636c6f636ba174a868616c666c696665cb4034000000000000a36c616d",
    "c0aa6d61785f64636c6f636bcb4014000000000000ae6f6e5f636c6f636b5f7265736574a36d",
    "6178a773657373696f6ec0ab73657373696f6e5f676170c0a6776569676874a177ab6d696e5f",
    "706572696f6473cb4008000000000000aa636f65665f657665727900aa656d69745f7369676d",
    "61c3ac656d69745f72657369645f7ac2ac656d69745f6d657472696373c2a9636f6e666f726d",
    "616cc0ae636f6e666f726d616c5f72617465c0af72657369645f7175616e74696c6573c0ad65",
    "6d69745f6175746f636f7272c2b272657369645f6175746f636f72725f6c6167c0aa656d6974",
    "5f6472696674c2ab64726966745f64656c7461c0af64726966745f7468726573686f6c64c0ac",
    "64726966745f616374696f6ec0ad656d69745f6176657261676564c2ab617665726167655f65",
    "7461c0ad656d69745f73656c6563746564c2a567726f7570a167de001fa46e616d65a172a56d",
    "6f64656c8aa474797065a865775f7269646765a57269646765cb3f847ae147ae147bac666561",
    "747572655f73657473c0ab7374616e64617264697a65c3ab72696467655f6465636179c2a563",
    "6f656630c0ae73657373696f6e5f736872696e6bc0ad6c6f6e675f68616c666c696665c0ab73",
    "6f6c76655f6576657279c0b76d61785f726f77735f6265747765656e5f736f6c766573c0a774",
    "61726765747391a179a8666561747572657392a27830a27831ad6164645f696e746572636570",
    "74c3a5636c6f636ba174a868616c666c696665cb4034000000000000a36c616dc0aa6d61785f",
    "64636c6f636bcb4014000000000000ae6f6e5f636c6f636b5f7265736574a36d6178a7736573",
    "73696f6ea473657373ab73657373696f6e5f676170a57265736574a6776569676874a177ab6d",
    "696e5f706572696f6473cb4008000000000000aa636f65665f657665727900aa656d69745f73",
    "69676d61c2ac656d69745f72657369645f7ac2ac656d69745f6d657472696373c2a9636f6e66",
    "6f726d616cc0ae636f6e666f726d616c5f72617465c0af72657369645f7175616e74696c6573",
    "c0ad656d69745f6175746f636f7272c2b272657369645f6175746f636f72725f6c6167c0aa65",
    "6d69745f6472696674c2ab64726966745f64656c7461c0af64726966745f7468726573686f6c",
    "64c0ac64726966745f616374696f6ec0ad656d69745f6176657261676564c2ab617665726167",
    "655f657461c0ad656d69745f73656c6563746564c2a567726f7570a167a67374617465739292",
    "92a1618ca5636c6f636b84aa707265765f636c6f636bcb404d000000000000ac707265765f73",
    "657373696f6ec0a770656e64696e67cb0000000000000000a773746172746564c3a66d6f6465",
    "6c739182ae736368656d615f76657273696f6e03a56d6f64656c81a64b616c6d616e87a36366",
    "678caa6e5f666561747572657302a96e5f7461726765747301ad6164645f696e746572636570",
    "74c3a5646563617981a848616c666c696665cb4034000000000000a868616c666c69666591cb",
    "4049000000000000a171c0a76f62735f766172c0a27030cb3ff0000000000000a77368617265",
    "5f70c2ab6d696e5f706572696f6473cb4008000000000000af7265766572745f68616c666c69",
    "666591cb7ff0000000000000ab7374616e64617264697a65c3a5737461747384a16b03a5775f",
    "73756dcb4021d10548a1848da16d93cb3ff0000000000000cbbf8987b750845516cb4026a40a",
    "51f6218aa16393cb0000000000000000cb3fb23c5d84f3b3bbcb3fe987834a12de2aa4626574",
    "619193cbc01449983db25747cb3fdd04bca1168734cbbfc60ce946a41472a1709199cb3fabb9",
    "7fd3e8c12bcb3f87c0607f388295cb3f5488242fc4c6fccb3f87c0607f388295cb3fa99db099",
    "ac2582cbbf69b9202717ae0acb3f5488242fc4c6fccbbf69b9202717ae0acb3f925b7e282824",
    "f0a47369673291cb3fdefab6c096d94ea47773696791cb401f3d0a9f6ce004a2776a91cb4020",
    "b36f24c18874a9726f77735f7365656e1ca972657369645f7661729191cb3fdefab6c096d94e",
    "a772657369645f779191cb401f3d0a9f6ce004a5647269667490a772657369645f7190a86175",
    "746f636f727290a76d65747269637390a9636f6e666f726d616c90a86c6173745f726f778ca4",
    "7072656491cbc016662f26f6fd37a5726573696491cbbfcd087b98321080a57369676d6191cb",
    "3fe7adf95f5751eea772657369645f7a91cbbfd39e091a0ecbf2a86175746f636f727290a76d",
    "65747269637390a9636f6e666f726d616c90a772657369645f7190a5647269667490a56e5f65",
    "666691cb4020f3a203d03e04ac6c616d5f73656c656374656490a4636f65669193cbc006f050",
    "19f95e97cb3ffb2e73481f9588cbbfc8afee05a20d9ea773756d6d6172798aa8726f77735f66",
    "65641eac726f77735f6c6561726e65641ab0726f77735f7a65726f5f77656967687400aa7765",
    "696768745f73756dcb4035000000000000a9636c6f636b5f6d696ecb0000000000000000a963",
    "6c6f636b5f6d6178cb404d000000000000af73657373696f6e5f6368616e67657300af636c6f",
    "636b5f6261636b776172647300a672657365747300a7636f6c756d6e739486a5636f756e741e",
    "a56e756c6c7300a46d65616ecb3f7d362e4c593c96a26d32cb4003bd502448ebc2a36d696ecb",
    "bfdfa66dd9042314a36d6178cb3fde6e454d6da8a686a5636f756e741ca56e756c6c7302a46d",
    "65616ecb4026e3b9b1842aeaa26d32cb403708d71506baada36d696ecb402407572427453ca3",
    "6d6178cb4029d8a1d7b0135286a5636f756e741ca56e756c6c7302a46d65616ecbc015b8bcb9",
    "ca63b6a26d32cb402ae5d8f23b83a6a36d696ecbc01b7bb0e5e3af19a36d6178cbc010a4fe94",
    "1caf0186a5636f756e741ea56e756c6c7300a46d65616ecb3fe8000000000000a26d32cb3ffe",
    "000000000000a36d696ecb3fe0000000000000a36d6178cb3ff000000000000092a1628ca563",
    "6c6f636b84aa707265765f636c6f636bcb404d800000000000ac707265765f73657373696f6e",
    "c0a770656e64696e67cb0000000000000000a773746172746564c3a66d6f64656c739182ae73",
    "6368656d615f76657273696f6e03a56d6f64656c81a64b616c6d616e87a36366678caa6e5f66",
    "6561747572657302a96e5f7461726765747301ad6164645f696e74657263657074c3a5646563",
    "617981a848616c666c696665cb4034000000000000a868616c666c69666591cb404900000000",
    "0000a171c0a76f62735f766172c0a27030cb3ff0000000000000a773686172655f70c2ab6d69",
    "6e5f706572696f6473cb4008000000000000af7265766572745f68616c666c69666591cb7ff0",
    "000000000000ab7374616e64617264697a65c3a5737461747384a16b03a5775f73756dcb4022",
    "b2828d9aaec8a16d93cb3ff0000000000000cb3f72f14a944ab838cb40279e972bcc97d8a163",
    "93cb0000000000000000cb3fb70e8c06cd3d20cb3fe89167e37541b4a4626574619193cbc012",
    "fdbb4fa10a66cb3fdc3bca30110f24cbbfc8abff6f5cdf11a1709199cb3fc869753d81e5b5cb",
    "3f8c3607d9bf9847cbbf9a69db3a01a8c1cb3f8c3607d9bf9847cb3fbbf97771955454cbbf60",
    "3337af53e48bcbbf9a69db3a01a8c1cbbf603337af53e48bcb3fb29bed365744c2a473696732",
    "91cb400840d5395b2fc5a47773696791cb4022969f6054e372a2776a91cb4023a5aa3bb5c7c0",
    "a9726f77735f7365656e1ca972657369645f7661729191cb400676f39dde48d6a77265736964",
    "5f779191cb4020f269d03fa01aa5647269667490a772657369645f7190a86175746f636f7272",
    "90a76d65747269637390a9636f6e666f726d616c90a86c6173745f726f778ca47072656491cb",
    "c01112b7f7cb986ea5726573696491cbbff241d50eeb92a4a57369676d6191cb3ffc073fab5d",
    "3475a772657369645f7a91cbbfe4d82059f7803ea86175746f636f727290a76d657472696373",
    "90a9636f6e666f726d616c90a772657369645f7190a5647269667490a56e5f65666691cb4021",
    "5c1e8dd0ae14ac6c616d5f73656c656374656490a4636f65669193cbc00140e1d12ebda8cb3f",
    "f784f5d4308851cbbfcc2849bceebbdba773756d6d6172798aa8726f77735f6665641eac726f",
    "77735f6c6561726e656415b0726f77735f7a65726f5f77656967687405aa7765696768745f73",
    "756dcb4036c00000000000a9636c6f636b5f6d696ecb3ff0000000000000a9636c6f636b5f6d",
    "6178cb404d800000000000af73657373696f6e5f6368616e67657300af636c6f636b5f626163",
    "6b776172647301a672657365747300a7636f6c756d6e739486a5636f756e741ea56e756c6c73",
    "00a46d65616ecbbf7293bf331ac610a26d32cb4004ce59fe8919d9a36d696ecbbfdff898216f",
    "5eeea36d6178cb3fde1c1b05026ccc86a5636f756e741ca56e756c6c7302a46d65616ecb4027",
    "4c5184441582a26d32cb4034b959c604873ca36d696ecb4024428b07237413a36d6178cb4029",
    "fad88da2485086a5636f756e741ba56e756c6c7303a46d65616ecbc016489d4ec6cb8ba26d32",
    "cb4031d5c3dd1843aba36d696ecbc01c7d823e79e102a36d6178cbc0100fd675bab2f686a563",
    "6f756e741ea56e756c6c7300a46d65616ecb3fe999999999999aa26d32cb4019333333333331",
    "a36d696ecb0000000000000000a36d6178cb3ff40000000000009292a1618ca5636c6f636b84",
    "aa707265765f636c6f636bcb404d000000000000ac707265765f73657373696f6ecf08d8ff07",
    "b578d149a770656e64696e67cb0000000000000000a773746172746564c3a66d6f64656c7391",
    "82ae736368656d615f76657273696f6e03a56d6f64656c81a7457752696467658ba36366678e",
    "aa6e5f666561747572657302a96e5f7461726765747301ad6164645f696e74657263657074c3",
    "a5646563617981a848616c666c696665cb4034000000000000a5726964676591cb3f847ae147",
    "ae147bac666561747572655f7365747390ab7374616e64617264697a65c3ab72696467655f64",
    "65636179c2a5636f656630c0ae73657373696f6e5f736872696e6bc0ad6c6f6e675f68616c66",
    "6c696665c0ab6d696e5f706572696f6473cb4008000000000000ab736f6c76655f6576657279",
    "cb3fd999999999999ab76d61785f726f77735f6265747765656e5f736f6c766573ceffffffff",
    "a3636f7687a16b03a5775f73756dcb401a2d7a8885922cab7072696f725f7363616c65cb3fd8",
    "406003b2ae5ea16d93cb3ff0000000000000cbbf838e70baf14fa4cb402688f89ebbe8d8a163",
    "99cb0000000000000000cb0000000000000000cb0000000000000000cb0000000000000000cb",
    "3fafcf9911390f8ccb3fb7836be37e2d8ccb0000000000000000cb3fb7836be37e2d8ccb3fe8",
    "c3eedead7a15af707265636973696f6e5f7072696f72cb0000000000000000af707265636973",
    "696f6e5f7363616c65cb3fada538ce11b67da2776a91cb4018f24e40c599f9a1729193cbc015",
    "aa75ce26011acb3fc44be60c26c0efcbc04ea6bc1cfbac9ea47773696791cb4011f1c46ced81",
    "10a47369673291cb3fee813185fb648fa4626574619193cb40009a079cbb4f52cb40056150b4",
    "56f489cbbfe5343bcffbf6aaa4736c6f77c0b1636c6f636b5f73696e63655f736f6c7665cb00",
    "00000000000000b0726f77735f73696e63655f736f6c766500ae736f6c76655f6661696c7572",
    "657300a9726f77735f7365656e1ca972657369645f7661729191cb3fee813185fb648fa77265",
    "7369645f779191cb4011f1c46ced8110a5647269667490a772657369645f7190a86175746f63",
    "6f727290a76d65747269637390a9636f6e666f726d616c90a86c6173745f726f778ca4707265",
    "6491cbc018136841fc8e20a5726573696491cb3fc89ea7c8800ca0a57369676d6190a7726573",
    "69645f7a90a86175746f636f727290a76d65747269637390a9636f6e666f726d616c90a77265",
    "7369645f7190a5647269667490a56e5f65666691cb4017c4f88ff3ae96ac6c616d5f73656c65",
    "6374656490a4636f65669193cb40009a079cbb4f52cb40056150b456f489cbbfe5343bcffbf6",
    "aaa773756d6d6172798aa8726f77735f6665641eac726f77735f6c6561726e65641ab0726f77",
    "735f7a65726f5f77656967687400aa7765696768745f73756dcb4035000000000000a9636c6f",
    "636b5f6d696ecb0000000000000000a9636c6f636b5f6d6178cb404d000000000000af736573",
    "73696f6e5f6368616e67657301af636c6f636b5f6261636b776172647300a672657365747301",
    "a7636f6c756d6e739486a5636f756e741ea56e756c6c7300a46d65616ecb3f7d362e4c593c96",
    "a26d32cb4003bd502448ebc2a36d696ecbbfdfa66dd9042314a36d6178cb3fde6e454d6da8a6",
    "86a5636f756e741ca56e756c6c7302a46d65616ecb4026e3b9b1842aeaa26d32cb403708d715",
    "06baada36d696ecb402407572427453ca36d6178cb4029d8a1d7b0135286a5636f756e741ca5",
    "6e756c6c7302a46d65616ecbc015b8bcb9ca63b6a26d32cb402ae5d8f23b83a6a36d696ecbc0",
    "1b7bb0e5e3af19a36d6178cbc010a4fe941caf0186a5636f756e741ea56e756c6c7300a46d65",
    "616ecb3fe8000000000000a26d32cb3ffe000000000000a36d696ecb3fe0000000000000a36d",
    "6178cb3ff000000000000092a1628ca5636c6f636b84aa707265765f636c6f636bcb404d8000",
    "00000000ac707265765f73657373696f6ecf08d8ff07b578d149a770656e64696e67cb000000",
    "0000000000a773746172746564c3a66d6f64656c739182ae736368656d615f76657273696f6e",
    "03a56d6f64656c81a7457752696467658ba36366678eaa6e5f666561747572657302a96e5f74",
    "61726765747301ad6164645f696e74657263657074c3a5646563617981a848616c666c696665",
    "cb4034000000000000a5726964676591cb3f847ae147ae147bac666561747572655f73657473",
    "90ab7374616e64617264697a65c3ab72696467655f6465636179c2a5636f656630c0ae736573",
    "73696f6e5f736872696e6bc0ad6c6f6e675f68616c666c696665c0ab6d696e5f706572696f64",
    "73cb4008000000000000ab736f6c76655f6576657279cb3fd999999999999ab76d61785f726f",
    "77735f6265747765656e5f736f6c766573ceffffffffa3636f7687a16b03a5775f73756dcb40",
    "1d887b0dbab772ab7072696f725f7363616c65cb3fd40ae9ddcc0b95a16d93cb3ff000000000",
    "0000cb3f81318a2a2c9490cb4027a06c1545deeca16399cb0000000000000000cb0000000000",
    "000000cb0000000000000000cb0000000000000000cb3fb5f0842f03221acb3f50f6de5c227c",
    "80cb0000000000000000cb3f50f6de5c227ca0cb3fe937bc6e50561aaf707265636973696f6e",
    "5f7072696f72cb0000000000000000af707265636973696f6e5f7363616c65cb3fab255df156",
    "d6f1a2776a91cb401bf7a0c866c68aa1729193cbc016ac1f62e7136bcb3fb80f523ee9720ccb",
    "c050ef73d27a725ea47773696791cb401929653864bc12a47369673291cb404b32de4e2b64a7",
    "a4626574619193cb4017d3ed65cdf904cb3ffa5b6b2f28e557cbbfef87079254c9dda4736c6f",
    "77c0b1636c6f636b5f73696e63655f736f6c7665cb0000000000000000b0726f77735f73696e",
    "63655f736f6c766500ae736f6c76655f6661696c7572657300a9726f77735f7365656e1ca972",
    "657369645f7661729191cb404b32de4e2b64a7a772657369645f779191cb401929653864bc12",
    "a5647269667490a772657369645f7190a86175746f636f727290a76d65747269637390a9636f",
    "6e666f726d616c90a86c6173745f726f778ca47072656491cbc018cec96ac8868ea572657369",
    "6491cb3fe95ce17a104bb8a57369676d6190a772657369645f7a90a86175746f636f727290a7",
    "6d65747269637390a9636f6e666f726d616c90a772657369645f7190a5647269667490a56e5f",
    "65666691cb401a4b40a55775e8ac6c616d5f73656c656374656490a4636f65669193cb4017d3",
    "ed65cdf904cb3ffa5b6b2f28e557cbbfef87079254c9dda773756d6d6172798aa8726f77735f",
    "6665641eac726f77735f6c6561726e656415b0726f77735f7a65726f5f77656967687405aa77",
    "65696768745f73756dcb4036c00000000000a9636c6f636b5f6d696ecb3ff0000000000000a9",
    "636c6f636b5f6d6178cb404d800000000000af73657373696f6e5f6368616e67657301af636c",
    "6f636b5f6261636b776172647301a672657365747301a7636f6c756d6e739486a5636f756e74",
    "1ea56e756c6c7300a46d65616ecbbf7293bf331ac610a26d32cb4004ce59fe8919d9a36d696e",
    "cbbfdff898216f5eeea36d6178cb3fde1c1b05026ccc86a5636f756e741ca56e756c6c7302a4",
    "6d65616ecb40274c5184441582a26d32cb4034b959c604873ca36d696ecb4024428b07237413",
    "a36d6178cb4029fad88da2485086a5636f756e741ba56e756c6c7303a46d65616ecbc016489d",
    "4ec6cb8ba26d32cb4031d5c3dd1843aba36d696ecbc01c7d823e79e102a36d6178cbc0100fd6",
    "75bab2f686a5636f756e741ea56e756c6c7300a46d65616ecb3fe999999999999aa26d32cb40",
    "19333333333331a36d696ecb0000000000000000a36d6178cb3ff4000000000000a8726f7773",
    "5f6665643c",
);

/// The stream the fixture was built from: every row a function of its index
/// alone (integer arithmetic into an exactly representable `f64`, no `sin`),
/// so any window is reproducible on any platform. Two interleaved groups;
/// `x1` is null every 17th row (the row is skipped), `y` every 13th (the row
/// is predicted, not learned), `w` is zero every 10th (the clock advances,
/// nothing is learned); the clock steps back once, at row 45 (to 42.5,
/// below its group's previous row), and the session turns over once, at
/// row 30, with the clock running on.
fn frame(start: usize, n: usize) -> DataFrame {
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }
    let mut t = Vec::with_capacity(n);
    let mut sess = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in start..start + n {
        let mut s = (i as u64) ^ 0x9E37_79B9_7F4A_7C15;
        let a = lcg(&mut s) - 0.5;
        let b = 10.0 + 3.0 * lcg(&mut s);
        let e = 0.1 * (lcg(&mut s) - 0.5);
        t.push(if i == 45 { 42.5 } else { i as f64 });
        sess.push(if i < 30 { "s0" } else { "s1" }.to_string());
        g.push(if i % 2 == 0 { "a" } else { "b" }.to_string());
        x0.push(a);
        x1.push((i % 17 != 3).then_some(b));
        y.push((i % 13 != 5).then_some(2.0 * a - 0.5 * b + 0.25 + e));
        w.push(if i % 10 == 7 {
            0.0
        } else {
            0.5 + (i % 4) as f64 * 0.25
        });
    }
    df!("t" => t, "sess" => sess, "g" => g, "x0" => x0, "x1" => x1, "y" => y, "w" => w).unwrap()
}

fn bytes() -> Vec<u8> {
    (0..SCHEMA3_STATE_HEX.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&SCHEMA3_STATE_HEX[i..i + 2], 16).unwrap())
        .collect()
}

/// The specs are recovered *from* the fixture: a file carries its own, and
/// restating them here would only test that two copies agree.
fn specs_from_fixture() -> Vec<Spec> {
    Bank::load_bytes(&bytes(), None)
        .expect("schema-3 fixture should load without spec expectations")
        .specs()
        .to_vec()
}

fn header(bytes: &[u8]) -> serde_json::Value {
    rmp_serde::from_slice(bytes).expect("a bank file is a msgpack map")
}

/// The bank the fixture is a snapshot of, built afresh in this build.
fn fresh() -> Bank {
    let mut bank = Bank::new(specs_from_fixture()).unwrap();
    for start in (0..60).step_by(20) {
        bank.fit_predict(&frame(start, 20)).unwrap();
    }
    bank
}

/// One frame per spec, unnested, so the comparison sees every field.
fn unnested(n: usize, cols: Vec<Column>) -> Vec<DataFrame> {
    let df = DataFrame::new(n, cols).unwrap();
    ["k", "r"]
        .iter()
        .map(|name| df.select([*name]).unwrap().unnest([*name], None).unwrap())
        .collect()
}

fn assert_same(want: &[DataFrame], got: &[DataFrame], what: &str) {
    for (w, g) in want.iter().zip(got) {
        assert!(w.equals_missing(g), "{what}:\n{w}\n{g}");
    }
}

/// Everything a bank reports about its state besides predictions: the
/// coefficients, the last learned rows, the data summary and the column
/// statistics, for every spec.
fn assert_same_state(want: &Bank, got: &Bank, what: &str) {
    for si in 0..want.specs().len() {
        assert_eq!(
            want.coef(si, None).unwrap(),
            got.coef(si, None).unwrap(),
            "{what}: spec {si}: coefficients"
        );
        let (wk, wc) = want.last_row(si, None).unwrap();
        let (gk, gc) = got.last_row(si, None).unwrap();
        assert_eq!(wk, gk, "{what}: spec {si}: last_row groups");
        assert!(
            wc.equals_missing(&gc),
            "{what}: spec {si}: last_row\n{wc:?}\n{gc:?}"
        );
        let (ws, gs) = (
            want.summary(si, None).unwrap(),
            got.summary(si, None).unwrap(),
        );
        assert!(
            ws.equals_missing(&gs),
            "{what}: spec {si}: summary\n{ws}\n{gs}"
        );
        let (wd, gd) = (
            want.describe(si, None).unwrap(),
            got.describe(si, None).unwrap(),
        );
        assert!(
            wd.equals_missing(&gd),
            "{what}: spec {si}: describe\n{wd}\n{gd}"
        );
    }
}

#[test]
fn the_fixture_is_what_it_says() {
    let h = header(&bytes());
    assert_eq!(h["format_version"], 2);
    assert_eq!(h["schema_version"], 3);
    let specs = specs_from_fixture();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].name, "k");
    assert_eq!(specs[1].name, "r");
    let json = serde_json::to_value(&specs).unwrap();
    assert_eq!(json, serde_json::to_value(generator_specs()).unwrap());
    assert_eq!(json[0]["model"]["type"], "kalman");
    assert_eq!(json[0]["model"]["standardize"], true);
    assert_eq!(json[1]["model"]["type"], "ew_ridge");
    assert_eq!(json[1]["session_gap"], "reset");
    // Every stream in the file carries a last row and a summary: that is
    // what makes it a schema-3 fixture rather than another schema-2 one.
    let states = h["states"].as_array().unwrap();
    assert_eq!(states.len(), 2);
    for per_spec in states {
        let streams = per_spec.as_array().unwrap();
        assert_eq!(streams.len(), 2, "two groups per spec");
        for pair in streams {
            let state = &pair[1];
            assert!(state["last_row"].is_object(), "{state}");
            let summary = &state["summary"];
            assert!(summary.is_object(), "{state}");
            assert_eq!(summary["rows_fed"], 30);
            assert_eq!(summary["columns"].as_array().unwrap().len(), 4);
        }
    }
}

#[test]
fn a_schema_3_state_file_loads_with_its_summary_and_last_row() {
    let bank = Bank::load_bytes(&bytes(), None);
    assert!(
        bank.is_ok(),
        "schema-3 state failed to load: {:?}",
        bank.err()
    );
    let specs = specs_from_fixture();
    assert!(Bank::load_bytes(&bytes(), Some(&specs)).is_ok());
    let restored = bank.unwrap();
    let want = fresh();
    assert_same_state(&want, &restored, "as loaded");
    // The numbers the fixture pins, so a silent change in what is counted
    // cannot hide behind a fresh bank changing the same way.
    let s = restored.summary(1, None).unwrap();
    let col = |name: &str| s.column(name).unwrap().clone();
    assert_eq!(col("group").str().unwrap().get(0), Some("a"));
    let u = |name: &str| col(name).u64().unwrap().to_vec();
    assert_eq!(u("rows_fed"), [Some(30), Some(30)]);
    // `x1` is null at rows 3, 20, 37, 54: two even, two odd.
    assert_eq!(u("rows_skipped"), [Some(2), Some(2)]);
    assert_eq!(u("rows_processed"), [Some(28), Some(28)]);
    // `w` is zero at rows 7, 17, 27, 37, 47, 57: all odd, and 37 is skipped.
    assert_eq!(u("rows_zero_weight"), [Some(0), Some(5)]);
    // `y` is null at rows 5, 18, 31, 44, 57: two even (accepted, not
    // learned), and of the odd ones 57 is already a zero weight.
    assert_eq!(u("rows_learned"), [Some(26), Some(21)]);
    assert_eq!(u("session_changes"), [Some(1), Some(1)]);
    assert_eq!(u("resets"), [Some(1), Some(1)]);
    assert_eq!(u("clock_backwards"), [Some(0), Some(1)]);
    let f = |name: &str| col(name).f64().unwrap().to_vec();
    assert_eq!(f("clock_min"), [Some(0.0), Some(1.0)]);
    assert_eq!(f("clock_max"), [Some(58.0), Some(59.0)]);
    assert_eq!(f("last_clock"), [Some(58.0), Some(59.0)]);
    // `k` has no session: the step back is a step back.
    let k = restored.summary(0, None).unwrap();
    let ku = |name: &str| k.column(name).unwrap().u64().unwrap().to_vec();
    assert_eq!(ku("session_changes"), [Some(0), Some(0)]);
    assert_eq!(ku("resets"), [Some(0), Some(0)]);
    assert_eq!(ku("clock_backwards"), [Some(0), Some(1)]);
    let d = restored.describe(1, None).unwrap();
    assert_eq!(d.height(), 8, "two groups x (x0, x1, y, w)");
    let roles = d.column("role").unwrap().str().unwrap();
    let roles: Vec<&str> = (0..roles.len()).map(|i| roles.get(i).unwrap()).collect();
    assert_eq!(roles, ["feature", "feature", "target", "weight"].repeat(2));
}

#[test]
fn a_schema_3_state_reports_no_kish_size_and_no_target_moments() {
    // Task 38 (E45) added `Sum w^2` and the per-target moments. A schema-3
    // file has neither, and neither can be replayed from what it does have:
    // a `Q` accumulated from the resume point against a `W` from the whole
    // stream reports an effective size too large by the length of the
    // history. So such a state reports `None` for the rest of its life --
    // and keeps streaming, and keeps re-saving the same bytes.
    let mut restored = Bank::load_bytes(&bytes(), None).unwrap();
    let g = &restored.gram(1, None).unwrap()[0];
    assert_eq!(g.n_kish, None, "no Sum w^2 in a schema-3 file");
    assert_eq!(g.target_means, None);
    assert_eq!(g.target_vars, None);
    assert_eq!(g.target_n_kish, None);
    // Not merely absent at load: absent for good.
    restored.fit_predict(&frame(60, 20)).unwrap();
    let g = &restored.gram(1, None).unwrap()[0];
    assert_eq!(g.n_kish, None, "a partial Sum w^2 would be worse than none");
    assert_eq!(g.target_vars, None);
    // The moments it does have are still the moments a fresh bank has.
    let mut want = fresh();
    want.fit_predict(&frame(60, 20)).unwrap();
    let w = &want.gram(1, None).unwrap()[0];
    assert_eq!(g.means, w.means);
    assert_eq!(g.comoments, w.comoments);
    assert_eq!(g.cross_moments, w.cross_moments);
    assert_eq!(g.target_weights, w.target_weights);
    // And a bank that never left the build reports all of it.
    assert!(w.n_kish.is_some());
    assert!(w.target_vars.is_some());
}

#[test]
fn a_schema_3_state_continues_the_stream_to_the_bit() {
    let specs = specs_from_fixture();
    let mut want = fresh();
    let mut restored = Bank::load_bytes(&bytes(), Some(&specs)).unwrap();
    let next = frame(60, 40);
    let w = unnested(next.height(), want.fit_predict(&next).unwrap());
    let g = unnested(next.height(), restored.fit_predict(&next).unwrap());
    assert_same(
        &w,
        &g,
        "a loaded schema-3 state did not continue identically",
    );
    assert_same_state(&want, &restored, "after 40 more rows");
    let probe = frame(100, 10);
    let w = unnested(probe.height(), want.predict(&probe).unwrap());
    let g = unnested(probe.height(), restored.predict(&probe).unwrap());
    assert_same(&w, &g, "predict differs");
    assert_same_state(&want, &restored, "after predict");
}

#[test]
fn a_schema_3_state_re_saves_byte_identically() {
    let restored = Bank::load_bytes(&bytes(), None).unwrap();
    let again = restored.save_bytes().unwrap();
    let h = header(&again);
    assert_eq!(h["format_version"], 2, "the envelope did not move");
    assert_eq!(h["schema_version"], online_core::SCHEMA_VERSION);
    let fixture = header(&bytes());
    if h["schema_version"] == fixture["schema_version"]
        && h["package_version"] == fixture["package_version"]
    {
        // Same writer: loading and saving is the identity, byte for byte.
        assert_eq!(again, bytes(), "the same writer re-saved different bytes");
    } else {
        // A later writer: the file must have moved to the current layout
        // and version, and nothing else. (When the layout changes, this
        // branch is the upgrade test `state_schema2.rs` has for schema 2.)
        assert_ne!(again, bytes(), "saving should write the current schema");
    }
    // Either way the re-saved file is the same bank.
    let reloaded = Bank::load_bytes(&again, None).unwrap();
    assert_same_state(&restored, &reloaded, "after re-save");
    assert_same_state(&fresh(), &reloaded, "after re-save, against fresh");
}

/// The specs the fixture was generated from, as written; the file carries
/// its own copy, and `the_fixture_is_what_it_says` checks they agree.
fn generator_specs() -> Vec<Spec> {
    serde_json::from_str(
        r#"[
            {"name": "k", "model": {"type": "kalman", "coef_halflife": 50.0,
             "standardize": true},
             "targets": ["y"], "features": ["x0", "x1"], "clock": "t",
             "halflife": 20.0, "max_dclock": 5.0, "weight": "w", "group": "g",
             "min_periods": 3.0, "emit_sigma": true},
            {"name": "r", "model": {"type": "ew_ridge", "ridge": 0.01,
             "standardize": true},
             "targets": ["y"], "features": ["x0", "x1"], "clock": "t",
             "session": "sess", "session_gap": "reset",
             "halflife": 20.0, "max_dclock": 5.0, "weight": "w", "group": "g",
             "min_periods": 3.0}
        ]"#,
    )
    .unwrap()
}

/// Generated the fixture: `cargo test -p online-polars --test state_schema3
/// print_fixture -- --ignored --nocapture`. Kept for the record; do not
/// regenerate the constant from a later layout.
#[test]
#[ignore]
fn print_fixture() {
    let mut bank = Bank::new(generator_specs()).unwrap();
    for start in (0..60).step_by(20) {
        bank.fit_predict(&frame(start, 20)).unwrap();
    }
    let bytes = bank.save_bytes().unwrap();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    println!("HEX_BEGIN");
    for line in hex.as_bytes().chunks(76) {
        println!("    \"{}\",", std::str::from_utf8(line).unwrap());
    }
    println!("HEX_END");
}
