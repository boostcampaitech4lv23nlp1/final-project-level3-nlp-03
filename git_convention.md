## Commit Convention.

**커밋 메세지 스타일 가이드**

### Commit Message Structure.

기본적으로 커밋 메세지는 아래와 같이 구성한다.

```bash
type: subject (#이슈번호)
```

**Git commit message 참고할 블로그.**

[좋은 git commit 메시지를 위한 영어 사전](https://blog.ull.im/engineering/2019/03/10/logs-on-git.html)

### Commit type

- `feat`: 새로운 기능 추가
- `fix`: 버그 수정, Simplify
- `docs`: 문서 수정, 주석 수정
- `delete`: 삭제(remove) 
- `style`: 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우
- `refactor`: 코드 리펙토링
- `test`: 테스트 코드, 리펙토링 테스트 코드 추가

### Subject

제목은 50자를 넘기지 않고, 첫 명령어에만 대문자로 작성, 마지막에 마침표를 붙이지 않는다.

과거시제를 사용하지 않고 명령어로 작성한다.

- “Fixed” → “Fix”
- “Added” → “Add”

------

## PR convention.

```bash
## 개요
`어떤 이유에서 이 PR을 시작하게 됐는지에 대한 히스토리를 남겨주세요.`

## 작업사항
`해당 이슈사항을 해결하기 위해 어떤 작업을 했는지 남겨주세요.`

## 로직
`어떤 목적을 가지고 딥러닝 코드를 작성했는지 간략히 써주세요.`

## Resolved
`해결한 Issue 번호를 적어주세요.`
```

## issue convention.

```bash
## 목표
`어떤 목표를 가지고 작업을 진행하는지 남겨주세요.`

## 세부사항
`어떤 세부사항이 예상되는지 작성해주세요.`        
or
## Target 
`어떤 부분을 수정하거나 추가가 할 것인지 작성해주세요.`        

## 세부사항
`어떤 세부사항을 수정/추가할 것인지 작성해주세요.`       
```

## Discussions convention.

##### Discussion Message Structure.

기본적으로 Discussion은 다음을 포함하여 게시한다.

```bash
type: Title (#이슈번호)

Current Status 

Analysis Result

Conculsion

Disscusion about ? (Optional)
```

##### Commit type

- `Ideas`: 논문 리스트, 방향 제시, 아이디어
- `Polls`: 수정 제안, ex) Add convention rule 
- `Q&A`: 질의
- `Show & Tell`: 결과 분석 결과 공유
- `General`: 그 외 