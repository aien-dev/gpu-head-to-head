import test from 'node:test';
import assert from 'node:assert/strict';
import {compareTokens,uniqueRows,findObservation} from './token-audit.mjs';
test('short prefixes and extra tokens cannot become identical',()=>{
    assert.equal(compareTokens([10,20],[10]).identical,false);
    assert.equal(compareTokens([10],[10,20]).first_div,1);
    assert.equal(compareTokens([10,20],[10,20]).identical,true);
    assert.equal(compareTokens([10,20],[10,21]).token_b,21);
});
test('a different token or divergence position cannot borrow an old tie',()=>{
    const records=[{prompt_id:'p',first_div:1,oracle_tok:20,arm_tok:21,oracle_len:2,logit_gap:0}];
    assert(findObservation(records,'p',compareTokens([10,20],[10,21])));
    assert.equal(findObservation(records,'p',compareTokens([10,20],[10,22])),null);
    assert.equal(findObservation(records,'p',compareTokens([20,10],[21,10])),null);
    assert.equal(findObservation(records,'other',compareTokens([10,20],[10,21])),null);
    assert.equal(findObservation(records,'p',compareTokens([10,20],[10])),null);
});
test('invalid IDs and duplicated/missing prompts fail closed',()=>{
    for(const ids of [[-1],[1.5],['1'],null]) assert.throws(()=>compareTokens(ids,[]));
    assert.throws(()=>uniqueRows([{prompt_id:'p'},{prompt_id:'p'}]));
    assert.throws(()=>uniqueRows([{}]));
});
